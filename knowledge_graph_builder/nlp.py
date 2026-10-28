from __future__ import annotations

import json
import hashlib
import math
import random
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - dependency is reported at use time
    torch = None
    nn = None


@dataclass
class DatasetExample:
    text: str
    ner_tokens: List[str] = field(default_factory=list)
    ner_tags: List[str] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("NLP training requires PyTorch. Install project dependencies with `pip install -r requirements.txt`.")


def _resolve_device(preference: str) -> Any:
    _require_torch()
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(preference)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; set device to `auto` or `cpu`.")
    return device


def split_dataset(
    examples: Sequence[DatasetExample],
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> Dict[str, List[DatasetExample]]:
    explicit_splits = {"train", "validation", "test"}
    split_values = [example.metadata.get("split") for example in examples if "split" in example.metadata]
    if split_values:
        if len(split_values) != len(examples) or set(split_values) != explicit_splits:
            raise ValueError("Explicit dataset splits must label every row and include train, validation, and test.")
        fixed = {
            split: [example for example in examples if example.metadata.get("split") == split]
            for split in ("train", "validation", "test")
        }
        if not all(fixed.values()):
            raise ValueError("Each explicit dataset split must contain examples for the selected task.")
        return fixed
    partitions = split_indices(len(examples), seed, ratios)
    return {key: [examples[index] for index in values] for key, values in partitions.items()}


def split_indices(
    size: int,
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> Dict[str, List[int]]:
    if size < 3:
        raise ValueError("At least 3 examples are required to create train, validation, and test splits.")
    if len(ratios) != 3 or any(not math.isfinite(ratio) or ratio <= 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("Split ratios must be three positive values summing to 1.")
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    train_size = max(1, int(len(indices) * ratios[0]))
    validation_size = max(1, int(len(indices) * ratios[1]))
    if train_size + validation_size >= len(indices):
        train_size = len(indices) - 2
        validation_size = 1
    return {
        "train": indices[:train_size],
        "validation": indices[train_size : train_size + validation_size],
        "test": indices[train_size + validation_size :],
    }


def dataset_digest(rows: Sequence[Any]) -> str:
    serialized = []
    for row in rows:
        if isinstance(row, DatasetExample):
            serialized.append(
                {
                    "text": row.text,
                    "ner_tokens": row.ner_tokens,
                    "ner_tags": row.ner_tags,
                    "relations": row.relations,
                    "metadata": row.metadata,
                }
            )
        else:
            serialized.append(row)
    content = json.dumps(serialized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def classification_metrics(expected: Sequence[str], predicted: Sequence[str]) -> Dict[str, Any]:
    if len(expected) != len(predicted):
        raise ValueError("Expected and predicted labels must have equal lengths.")
    labels = sorted(set(expected) | set(predicted))
    per_class: Dict[str, Dict[str, float]] = {}
    for label in labels:
        tp = sum(actual == label and guess == label for actual, guess in zip(expected, predicted))
        fp = sum(actual != label and guess == label for actual, guess in zip(expected, predicted))
        fn = sum(actual == label and guess != label for actual, guess in zip(expected, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        }
    accuracy = sum(actual == guess for actual, guess in zip(expected, predicted)) / len(expected) if expected else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": sum(scores["f1"] for scores in per_class.values()) / len(labels) if labels else 0.0,
        "per_class": per_class,
        "count": len(expected),
    }


class _RelationClassifier(nn.Module if nn is not None else object):
    def __init__(self, vocabulary_size: int, num_labels: int, embedding_dim: int = 64) -> None:
        _require_torch()
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, num_labels),
        )

    def forward(self, token_ids: Any) -> Any:
        mask = token_ids.ne(0).unsqueeze(-1)
        embeddings = self.embedding(token_ids)
        pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.classifier(pooled)


def _ner_token_features(tokens: Sequence[str], index: int) -> List[str]:
    token = tokens[index]
    normalized = token.lower()
    previous = tokens[index - 1] if index else "<bos>"
    following = tokens[index + 1] if index + 1 < len(tokens) else "<eos>"
    return [
        f"token={normalized}",
        f"prefix2={normalized[:2]}",
        f"prefix3={normalized[:3]}",
        f"suffix2={normalized[-2:]}",
        f"suffix3={normalized[-3:]}",
        f"case={'upper' if token.isupper() else 'title' if token.istitle() else 'lower' if token.islower() else 'mixed'}",
        f"shape={'alpha' if token.isalpha() else 'digit' if token.isdigit() else 'alnum' if token.isalnum() else 'punct'}",
        f"has_digit={any(char.isdigit() for char in token)}",
        f"previous={previous.lower()}",
        f"next={following.lower()}",
        f"previous_case={'title' if previous.istitle() else 'other'}",
        f"next_case={'title' if following.istitle() else 'other'}",
    ]


class _LocalNERClassifier(nn.Module if nn is not None else object):
    def __init__(self, feature_count: int, num_labels: int, embedding_dim: int = 32) -> None:
        _require_torch()
        super().__init__()
        self.embedding = nn.Embedding(feature_count, embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, num_labels),
        )

    def forward(self, feature_ids: Any) -> Any:
        token_features = self.embedding(feature_ids).mean(dim=2)
        return self.classifier(token_features)


def _local_ner_ids(tokens: Sequence[str], feature_to_id: Dict[str, int]) -> List[List[int]]:
    return [
        [feature_to_id.get(feature, 0) for feature in _ner_token_features(tokens, index)]
        for index in range(len(tokens))
    ]


class EntityRecognizer:
    """Transformer NER inference using a locally saved, fine-tuned checkpoint."""

    def __init__(self, checkpoint: str | Path) -> None:
        _require_torch()
        self.checkpoint = Path(checkpoint)
        metadata = json.loads((self.checkpoint / "metadata.json").read_text(encoding="utf-8"))
        self.id_to_label = {int(key): value for key, value in metadata["id_to_label"].items()}
        self.architecture = metadata.get("architecture", "pretrained-transformer-token-classification")
        known_tokens = metadata.get("known_tokens")
        self.known_tokens = set(known_tokens) if isinstance(known_tokens, list) else None
        if self.architecture == "local-context-feature-classifier":
            self.feature_to_id = metadata["feature_to_id"]
            self.model = _LocalNERClassifier(
                len(self.feature_to_id),
                len(self.id_to_label),
                int(metadata["embedding_dim"]),
            )
            self.model.load_state_dict(
                torch.load(self.checkpoint / "weights.pt", map_location="cpu", weights_only=True)
            )
            self.tokenizer = None
        else:
            try:
                from transformers import AutoModelForTokenClassification, AutoTokenizer
            except ImportError as error:
                raise RuntimeError("Transformer NER requires `transformers`; install project dependencies.") from error
            self.feature_to_id = {}
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, local_files_only=True)
            self.model = AutoModelForTokenClassification.from_pretrained(self.checkpoint, local_files_only=True)
        self.model.eval()

    def predict_tags(self, words: Sequence[str]) -> List[str]:
        if self.architecture == "local-context-feature-classifier":
            if not words:
                return []
            inputs = torch.tensor([_local_ner_ids(words, self.feature_to_id)], dtype=torch.long)
            with torch.no_grad():
                predictions = self.model(inputs).argmax(dim=-1)[0].tolist()
            labels = [self.id_to_label[index] for index in predictions]
            if self.known_tokens is not None:
                labels = [
                    label if word.lower() in self.known_tokens else "O"
                    for word, label in zip(words, labels)
                ]
            return labels
        encoded = self.tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = self.model(**encoded).logits[0]
        labels = ["O"] * len(words)
        seen_words = set()
        for token_index, word_index in enumerate(encoded.word_ids()):
            if word_index is None or word_index in seen_words:
                continue
            seen_words.add(word_index)
            labels[word_index] = self.id_to_label[int(logits[token_index].argmax().item())]
        return labels

    def infer(self, text: str) -> List[Dict[str, Any]]:
        spans = list(re.finditer(r"\S+", text))
        words = [match.group(0) for match in spans]
        if not words:
            return []
        word_labels = self.predict_tags(words)
        return _bio_entities(word_labels, words, [(match.start(), match.end()) for match in spans])


def _bio_entities(
    labels: Sequence[str], words: Sequence[str], offsets: Sequence[Tuple[int, int]]
) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []
    start = None
    active_label = None
    for index, tag in enumerate([*labels, "O"]):
        prefix, _, entity_label = tag.partition("-")
        continues = prefix == "I" and entity_label == active_label
        begins = prefix == "B" or (prefix not in {"O", "I"} and tag != "O")
        if start is not None and not continues:
            end_index = index - 1
            entities.append(
                {
                    "text": " ".join(words[start : end_index + 1]),
                    "label": active_label,
                    "start": offsets[start][0],
                    "end": offsets[end_index][1],
                }
            )
            start = None
            active_label = None
        if begins or (prefix == "I" and start is None):
            start = index
            active_label = entity_label if prefix in {"B", "I"} else tag
    return entities


class RelationExtractor:
    """Trainable PyTorch relation classifier; it does not invent entity spans."""

    def __init__(self, checkpoint: str | Path) -> None:
        _require_torch()
        self.checkpoint = Path(checkpoint)
        metadata = json.loads((self.checkpoint / "metadata.json").read_text(encoding="utf-8"))
        self.vocabulary = metadata["vocabulary"]
        self.id_to_label = {int(key): value for key, value in metadata["id_to_label"].items()}
        self.label_to_id = {value: key for key, value in self.id_to_label.items()}
        self.max_length = int(metadata["max_length"])
        self.model = _RelationClassifier(len(self.vocabulary), len(self.id_to_label), metadata["embedding_dim"])
        self.model.load_state_dict(torch.load(self.checkpoint / "weights.pt", map_location="cpu", weights_only=True))
        self.model.eval()

    def predict(self, text: str, subject: Dict[str, int], obj: Dict[str, int]) -> Dict[str, Any]:
        _validate_span(text, subject, "subject")
        _validate_span(text, obj, "object")
        token_ids = torch.tensor([_relation_token_ids(text, subject, obj, self.vocabulary, self.max_length)])
        with torch.no_grad():
            logits = self.model(token_ids)[0]
            probabilities = torch.softmax(logits, dim=-1)
            label_id = int(logits.argmax().item())
        return {
            "label": self.id_to_label[label_id],
            "score": float(probabilities[label_id].item()),
            "score_type": "uncalibrated_softmax_probability",
            "subject": text[subject["start"] : subject["end"]],
            "object": text[obj["start"] : obj["end"]],
        }


class EntityResolver:
    """Conservative reversible duplicate-mention suggestions; never merges automatically."""

    def __init__(self, threshold: float = 0.8) -> None:
        self.threshold = threshold

    @staticmethod
    def normalize(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    @staticmethod
    def similarity(left: str, right: str) -> float:
        left_norm = EntityResolver.normalize(left)
        right_norm = EntityResolver.normalize(right)
        if left_norm == right_norm:
            return 1.0
        if not left_norm or not right_norm:
            return 0.0
        return 0.0

    def resolve(self, mentions: Iterable[str]) -> List[Dict[str, Any]]:
        records = list(mentions)
        suggestions: List[Dict[str, Any]] = []
        for index, mention in enumerate(records):
            candidates = [(self.similarity(mention, old), old) for old in records[:index]]
            score, candidate = max(candidates, default=(0.0, None))
            suggestions.append(
                {"mention": mention, "candidate": candidate if score >= self.threshold else None, "score": score, "merge": False}
            )
        return suggestions


def _validate_span(text: str, span: Dict[str, int], name: str) -> None:
    if (
        not isinstance(span, dict)
        or not isinstance(span.get("start"), int)
        or isinstance(span.get("start"), bool)
        or not isinstance(span.get("end"), int)
        or isinstance(span.get("end"), bool)
    ):
        raise ValueError(f"{name} must contain integer start and end character offsets.")
    if span["start"] < 0 or span["end"] <= span["start"] or span["end"] > len(text):
        raise ValueError(f"{name} span is outside the text or empty.")


def _relation_tokens(text: str, subject: Dict[str, int], obj: Dict[str, int]) -> List[str]:
    marked = (
        text[: subject["start"]]
        + " SUBJ "
        + text[subject["start"] : subject["end"]]
        + " "
        + text[subject["end"] : obj["start"]]
        + " OBJ "
        + text[obj["start"] : obj["end"]]
        + " "
        + text[obj["end"] :]
        if subject["end"] <= obj["start"]
        else text[: obj["start"]]
        + " OBJ "
        + text[obj["start"] : obj["end"]]
        + " "
        + text[obj["end"] : subject["start"]]
        + " SUBJ "
        + text[subject["start"] : subject["end"]]
        + " "
        + text[subject["end"] :]
    )
    return re.findall(r"\w+|[^\w\s]", marked.lower())


def _relation_token_ids(
    text: str, subject: Dict[str, int], obj: Dict[str, int], vocabulary: Dict[str, int], max_length: int
) -> List[int]:
    return [vocabulary.get(token, 1) for token in _relation_tokens(text, subject, obj)[:max_length]]


def _relation_record(example: DatasetExample) -> Tuple[str, Dict[str, int], Dict[str, int], str]:
    if len(example.relations) != 1:
        raise ValueError("Each relation-training example must contain exactly one relation annotation.")
    relation = example.relations[0]
    subject, obj, label = relation.get("subject"), relation.get("object"), relation.get("label")
    _validate_span(example.text, subject, "subject")
    _validate_span(example.text, obj, "object")
    if subject["start"] < obj["end"] and obj["start"] < subject["end"]:
        raise ValueError("Subject and object spans must not overlap.")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("Relation annotations require a non-empty label.")
    return example.text, subject, obj, label.strip()


class NLPTrainingPipeline:
    """Dataset validation, trainable relation classification, and transformer NER fine-tuning."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.random_seed = int(self.config.get("random_seed", 42))
        self.epochs = int(self.config.get("epochs", 3))
        self.batch_size = int(self.config.get("batch_size", 8))
        self.learning_rate = float(self.config.get("learning_rate", 2e-3))
        self.dataset_path = self.config.get("dataset_path")
        self.checkpoint_dir = Path(self.config.get("checkpoint_dir", "checkpoints/nlp"))
        self.max_length = int(self.config.get("max_length", 128))
        self.embedding_dim = int(self.config.get("embedding_dim", 64))
        self.pretrained_model = self.config.get("pretrained_model", "distilbert/distilbert-base-uncased")
        self.device_preference = self.config.get("device", "auto")
        self.synthetic_data = bool(self.config.get("synthetic_data", False))
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive integers.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be a finite positive number.")
        if self.max_length < 1 or self.embedding_dim < 1:
            raise ValueError("max_length and embedding_dim must be positive integers.")

    def load_dataset(self, path: Optional[str | Path] = None) -> List[DatasetExample]:
        requested_path = path or self.dataset_path
        if not requested_path:
            raise FileNotFoundError("Provide `dataset_path`; no training dataset is configured.")
        source = Path(requested_path)
        if not source.exists():
            raise FileNotFoundError(f"NLP dataset was not found: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, (dict, list)):
            raise ValueError("NLP dataset must be a JSON object or array.")
        rows = payload if isinstance(payload, list) else payload.get("examples")
        self.synthetic_data = bool(payload.get("synthetic", False)) if isinstance(payload, dict) else False
        if not isinstance(rows, list) or not rows:
            raise ValueError("NLP dataset must contain a non-empty `examples` array.")
        examples = [
            DatasetExample(
                text=row.get("text", ""),
                ner_tokens=row.get("ner_tokens", []),
                ner_tags=row.get("ner_tags", []),
                relations=row.get("relations", []),
                metadata=row.get("metadata", {}),
            )
            for row in rows
            if isinstance(row, dict)
        ]
        if len(examples) != len(rows):
            raise ValueError("Every dataset example must be a JSON object.")
        for index, example in enumerate(examples):
            if not isinstance(example.text, str) or not example.text.strip():
                raise ValueError(f"Example {index} requires non-empty text.")
            if not isinstance(example.ner_tokens, list) or not isinstance(example.ner_tags, list):
                raise ValueError(f"Example {index} NER tokens and tags must be arrays.")
            if bool(example.ner_tokens) != bool(example.ner_tags) or len(example.ner_tokens) != len(example.ner_tags):
                raise ValueError(f"Example {index} must have equal-length `ner_tokens` and `ner_tags` arrays.")
            if any(not isinstance(token, str) or not token for token in example.ner_tokens):
                raise ValueError(f"Example {index} NER tokens must be non-empty strings.")
            self._validate_bio_tags(example.ner_tags, index)
            if not isinstance(example.relations, list):
                raise ValueError(f"Example {index} relations must be an array.")
            if not isinstance(example.metadata, dict):
                raise ValueError(f"Example {index} metadata must be an object.")
            if example.relations and len(example.relations) != 1:
                raise ValueError(f"Example {index} must have exactly one relation; duplicate the row for additional relations.")
            for relation in example.relations:
                if not isinstance(relation, dict):
                    raise ValueError(f"Example {index} relation annotation must be an object.")
                _relation_record(DatasetExample(example.text, relations=[relation]))
        split_values = [example.metadata.get("split") for example in examples if "split" in example.metadata]
        if split_values and (
            len(split_values) != len(examples)
            or set(split_values) != {"train", "validation", "test"}
        ):
            raise ValueError("Explicit dataset splits must label every row and include train, validation, and test.")
        return examples

    @staticmethod
    def _validate_bio_tags(tags: Sequence[str], example_index: int) -> None:
        active_type = None
        for tag in tags:
            if not isinstance(tag, str):
                raise ValueError(f"Example {example_index} NER tags must be strings.")
            if tag == "O":
                active_type = None
                continue
            prefix, separator, entity_type = tag.partition("-")
            if not separator or prefix not in {"B", "I"} or not entity_type:
                raise ValueError(f"Example {example_index} has invalid BIO tag {tag!r}.")
            if prefix == "I" and entity_type != active_type:
                raise ValueError(f"Example {example_index} has an I-tag without a matching preceding B/I tag.")
            active_type = entity_type

    def validate_dataset(self, dataset: Optional[Sequence[DatasetExample]] = None, task: str = "relation") -> Dict[str, Any]:
        examples = list(dataset) if dataset is not None else self.load_dataset()
        if task not in {"relation", "ner"}:
            raise ValueError("Task must be `relation` or `ner`.")
        if task == "ner":
            selected = [example for example in examples if example.ner_tokens]
            if not selected:
                raise ValueError("NER training requires examples with `ner_tokens` and aligned `ner_tags`.")
        else:
            selected = [example for example in examples if example.relations]
            if not selected:
                raise ValueError("Relation training requires examples with text, subject/object spans, and relation labels.")
        split = split_dataset(selected, self.random_seed)
        return {key: len(value) for key, value in split.items()}

    def train(
        self,
        dataset: Optional[List[DatasetExample]] = None,
        task: str = "relation",
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        examples = dataset if dataset is not None else self.load_dataset()
        if task == "relation":
            return self.train_relation(examples, progress_callback)
        if task == "ner":
            return self.train_ner(examples, progress_callback)
        raise ValueError("Task must be `relation` or `ner`.")

    def _checkpoint_path(self, task: str) -> Path:
        path = self.checkpoint_dir / f"{task}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def train_relation(
        self,
        examples: Sequence[DatasetExample],
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        _require_torch()
        records = [_relation_record(example) for example in examples if example.relations]
        relation_examples = [
            DatasetExample(
                text=text,
                relations=[{"subject": subject, "object": obj, "label": label}],
                metadata=example.metadata,
            )
            for example in examples if example.relations
            for text, subject, obj, label in [_relation_record(example)]
        ]
        split = split_dataset(relation_examples, self.random_seed)
        if relation_examples and all(example.metadata.get("split") in {"train", "validation", "test"} for example in relation_examples):
            split_index_map = {
                name: [index for index, example in enumerate(relation_examples) if example.metadata.get("split") == name]
                for name in ("train", "validation", "test")
            }
        else:
            split_index_map = split_indices(len(relation_examples), self.random_seed)
        all_labels = sorted({record[3] for record in records})
        if len(all_labels) < 2:
            raise ValueError("Relation training requires at least two distinct labeled relation classes.")
        train_records = [_relation_record(example) for example in split["train"]]
        validation_records = [_relation_record(example) for example in split["validation"]]
        test_records = [_relation_record(example) for example in split["test"]]
        training_labels = sorted({record[3] for record in train_records})
        if len(training_labels) < 2:
            raise ValueError("Relation training requires at least two distinct relation labels in the training split.")
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        device = _resolve_device(self.device_preference)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(self.random_seed)
        vocabulary = {"<pad>": 0, "<unk>": 1}
        for text, subject, obj, _ in train_records:
            for token in _relation_tokens(text, subject, obj)[: self.max_length]:
                vocabulary.setdefault(token, len(vocabulary))
        label_to_id = {label: index for index, label in enumerate(training_labels)}

        def batch(rows: Sequence[Tuple[str, Dict[str, int], Dict[str, int], str]]) -> Tuple[Any, Any]:
            features = [_relation_token_ids(text, sub, obj, vocabulary, self.max_length) for text, sub, obj, _ in rows]
            width = max((len(tokens) for tokens in features), default=1)
            inputs = torch.tensor(
                [tokens + [0] * (width - len(tokens)) for tokens in features], dtype=torch.long, device=device
            )
            targets = torch.tensor(
                [label_to_id.get(label, 0) for _, _, _, label in rows], dtype=torch.long, device=device
            )
            return inputs, targets

        model = _RelationClassifier(len(vocabulary), len(label_to_id), self.embedding_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        train_losses: List[float] = []
        validation_losses: List[float] = []
        for _ in range(self.epochs):
            model.train()
            epoch_losses = []
            order = list(range(len(train_records)))
            random.shuffle(order)
            for offset in range(0, len(order), self.batch_size):
                rows = [train_records[index] for index in order[offset : offset + self.batch_size]]
                inputs, targets = batch(rows)
                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(model(inputs), targets)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().item()))
            train_losses.append(sum(epoch_losses) / len(epoch_losses))
            seen_validation = [row for row in validation_records if row[3] in label_to_id]
            if not seen_validation:
                raise ValueError("Validation split has no relation labels represented in the training split.")
            validation_losses.append(self._relation_loss(model, seen_validation, batch))
            if progress_callback:
                progress_callback(
                    f"epoch={len(train_losses)}/{self.epochs} train_loss={train_losses[-1]:.6f} "
                    f"validation_loss={validation_losses[-1]:.6f}"
                )

        checkpoint = self._checkpoint_path("relation")
        torch.save(model.state_dict(), checkpoint / "weights.pt")
        test_metrics = self._relation_metrics(model, test_records, batch, label_to_id)
        majority_label = max(training_labels, key=lambda label: sum(row[3] == label for row in train_records))
        baseline_metrics = classification_metrics(
            [row[3] for row in test_records],
            [majority_label] * len(test_records),
        )
        metadata = {
            "task": "relation",
            "architecture": "mean-pooled-token-embedding-mlp",
            "vocabulary": vocabulary,
            "id_to_label": {str(value): key for key, value in label_to_id.items()},
            "max_length": self.max_length,
            "embedding_dim": self.embedding_dim,
            "seed": self.random_seed,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "split_sizes": {key: len(rows) for key, rows in split.items()},
            "split_indices": split_index_map,
            "split_source": "dataset" if any(example.metadata.get("split") for example in relation_examples) else "seeded_random",
            "dataset_sha256": dataset_digest(records),
            "training_examples": len(train_records),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "synthetic_data": self.synthetic_data,
            "device": str(device),
            "train_loss": train_losses,
            "validation_loss": validation_losses,
            "evaluation": test_metrics,
            "baseline": {
                "name": "training-split majority class",
                "label": majority_label,
                "evaluation": baseline_metrics,
            },
        }
        (checkpoint / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return {
            "status": "completed",
            "task": "relation",
            "checkpoint": str(checkpoint),
            "train_loss": train_losses,
            "validation_loss": validation_losses,
            "evaluation": test_metrics,
            "baseline": metadata["baseline"],
            "split_sizes": metadata["split_sizes"],
            "synthetic_data": metadata["synthetic_data"],
            "interpretation": "Synthetic smoke-test data; these metrics are not meaningful model-performance estimates." if self.synthetic_data else None,
        }

    @staticmethod
    def evaluate_relation_checkpoint(checkpoint: str | Path, examples: Sequence[DatasetExample]) -> Dict[str, Any]:
        _require_torch()
        extractor = RelationExtractor(checkpoint)
        relation_examples = [
            DatasetExample(text=example.text, relations=example.relations, metadata=example.metadata)
            for example in examples if example.relations
        ]
        records = [_relation_record(example) for example in relation_examples]
        metadata = json.loads((Path(checkpoint) / "metadata.json").read_text(encoding="utf-8"))
        if dataset_digest(records) != metadata["dataset_sha256"]:
            raise ValueError("Evaluation dataset does not match the checkpoint's recorded dataset fingerprint.")
        split = split_dataset(relation_examples, int(metadata["seed"]))
        test_records = [_relation_record(example) for example in split["test"]]
        if not test_records:
            raise ValueError("The evaluation split contains no relation examples.")
        expected = [label for _, _, _, label in test_records]
        predicted = [
            extractor.predict(text, subject, obj)["label"]
            for text, subject, obj, _ in test_records
        ]
        return classification_metrics(expected, predicted)

    @staticmethod
    def _relation_loss(model: Any, rows: Sequence[Any], batch_fn: Any) -> float:
        model.eval()
        with torch.no_grad():
            inputs, targets = batch_fn(rows)
            return float(nn.functional.cross_entropy(model(inputs), targets).item())

    @staticmethod
    def _relation_metrics(model: Any, rows: Sequence[Any], batch_fn: Any, label_to_id: Dict[str, int]) -> Dict[str, Any]:
        if not rows:
            return classification_metrics([], [])
        model.eval()
        inputs, targets = batch_fn(rows)
        with torch.no_grad():
            predicted_ids = model(inputs).argmax(dim=-1).tolist()
        id_to_label = {value: key for key, value in label_to_id.items()}
        return classification_metrics(
            [row[3] for row in rows],
            [id_to_label[index] for index in predicted_ids],
        )

    def train_ner(
        self,
        examples: Sequence[DatasetExample],
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        _require_torch()
        if self.config.get("ner_architecture", "pretrained-transformer") == "local-context":
            return self.train_ner_local(examples, progress_callback)
        try:
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Transformer NER requires `transformers`; install project dependencies.") from error
        selected = [example for example in examples if example.ner_tokens]
        if not selected:
            raise ValueError("NER training requires examples with aligned `ner_tokens` and `ner_tags`.")
        for index, example in enumerate(selected):
            if len(example.ner_tokens) != len(example.ner_tags):
                raise ValueError("NER token/tag alignment mismatch.")
            self._validate_bio_tags(example.ner_tags, index)
        split = split_dataset(selected, self.random_seed)
        if selected and all(example.metadata.get("split") in {"train", "validation", "test"} for example in selected):
            split_index_map = {
                name: [index for index, example in enumerate(selected) if example.metadata.get("split") == name]
                for name in ("train", "validation", "test")
            }
        else:
            split_index_map = split_indices(len(selected), self.random_seed)
        label_names = sorted({"O"} | {label for example in selected for label in example.ner_tags})
        label_to_id = {label: index for index, label in enumerate(label_names)}
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
            model = AutoModelForTokenClassification.from_pretrained(
                self.pretrained_model, num_labels=len(label_to_id), id2label={i: label for label, i in label_to_id.items()},
                label2id=label_to_id, ignore_mismatched_sizes=True
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not load pretrained NER model `{self.pretrained_model}`. Check network access/cache or set `pretrained_model` to a local Hugging Face model directory."
            ) from error
        torch.manual_seed(self.random_seed)
        device = _resolve_device(self.device_preference)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(self.random_seed)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(self.config.get("learning_rate", 2e-5)))
        train_losses: List[float] = []
        validation_losses: List[float] = []
        for _ in range(self.epochs):
            model.train()
            losses = []
            order = list(range(len(split["train"])))
            random.Random(self.random_seed + len(train_losses)).shuffle(order)
            for index in order:
                example = split["train"][index]
                encoded = tokenizer(
                    example.ner_tokens, is_split_into_words=True, return_tensors="pt", truncation=True
                ).to(device)
                word_ids = encoded.word_ids()
                labels = []
                previous = None
                for word_id in word_ids:
                    labels.append(-100 if word_id is None or word_id == previous else label_to_id[example.ner_tags[word_id]])
                    previous = word_id
                targets = torch.tensor([labels], dtype=torch.long, device=device)
                optimizer.zero_grad()
                loss = model(**encoded, labels=targets).loss
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().item()))
            train_losses.append(sum(losses) / len(losses))
            validation_losses.append(self._ner_loss(model, tokenizer, split["validation"], label_to_id))
            if progress_callback:
                progress_callback(
                    f"epoch={len(train_losses)}/{self.epochs} train_loss={train_losses[-1]:.6f} "
                    f"validation_loss={validation_losses[-1]:.6f}"
                )
        checkpoint = self._checkpoint_path("ner")
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        metadata = {
            "task": "ner",
            "architecture": "pretrained-transformer-token-classification",
            "id_to_label": {str(index): label for label, index in label_to_id.items()},
            "seed": self.random_seed,
            "epochs": self.epochs,
            "pretrained_model": self.pretrained_model,
            "split_sizes": {key: len(rows) for key, rows in split.items()},
            "split_indices": split_index_map,
            "split_source": "dataset" if any(example.metadata.get("split") for example in selected) else "seeded_random",
            "dataset_sha256": dataset_digest(selected),
            "synthetic_data": self.synthetic_data,
            "device": str(device),
        }
        (checkpoint / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        evaluation = self.evaluate_ner(split["test"], checkpoint)
        baseline = self._ner_all_outside_baseline(split["test"])
        metadata["train_loss"] = train_losses
        metadata["validation_loss"] = validation_losses
        metadata["evaluation"] = evaluation
        metadata["baseline"] = baseline
        (checkpoint / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return {
            "status": "completed",
            "task": "ner",
            "checkpoint": str(checkpoint),
            "train_loss": train_losses,
            "validation_loss": validation_losses,
            "split_sizes": metadata["split_sizes"],
            "synthetic_data": metadata["synthetic_data"],
            "evaluation": evaluation,
            "baseline": baseline,
            "interpretation": "Synthetic smoke-test data; these metrics are not meaningful model-performance estimates." if self.synthetic_data else None,
        }

    def train_ner_local(
        self,
        examples: Sequence[DatasetExample],
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        _require_torch()
        selected = [example for example in examples if example.ner_tokens]
        if not selected:
            raise ValueError("NER training requires examples with aligned `ner_tokens` and `ner_tags`.")
        for index, example in enumerate(selected):
            if len(example.ner_tokens) != len(example.ner_tags):
                raise ValueError(f"NER token/tag alignment mismatch in example {index}.")
            self._validate_bio_tags(example.ner_tags, index)
        split = split_dataset(selected, self.random_seed)
        selected_indices = {id(example): index for index, example in enumerate(selected)}
        split_index_map = {
            name: [selected_indices[id(example)] for example in split[name]]
            for name in ("train", "validation", "test")
        }
        train_features = {
            feature
            for example in split["train"]
            for token_index in range(len(example.ner_tokens))
            for feature in _ner_token_features(example.ner_tokens, token_index)
        }
        feature_to_id = {"<unk>": 0}
        feature_to_id.update({feature: index + 1 for index, feature in enumerate(sorted(train_features))})
        known_tokens = sorted({
            token.lower()
            for example in split["train"]
            for token in example.ner_tokens
        })
        label_names = sorted({"O"} | {label for example in selected for label in example.ner_tags})
        label_to_id = {label: index for index, label in enumerate(label_names)}
        torch.manual_seed(self.random_seed)
        random.seed(self.random_seed)
        device = _resolve_device(self.device_preference)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(self.random_seed)
        embedding_dim = int(self.config.get("ner_embedding_dim", 32))
        model = _LocalNERClassifier(len(feature_to_id), len(label_to_id), embedding_dim).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config.get("learning_rate", 1e-2)),
        )
        def loss_for(rows: Sequence[DatasetExample], training: bool = False) -> float:
            model.train(training)
            losses = []
            for row in rows:
                token_features = _local_ner_ids(row.ner_tokens, feature_to_id)
                inputs = torch.tensor([token_features], dtype=torch.long, device=device)
                targets = torch.tensor(
                    [label_to_id[tag] for tag in row.ner_tags],
                    dtype=torch.long,
                    device=device,
                )
                if training:
                    optimizer.zero_grad()
                with torch.set_grad_enabled(training):
                    logits = model(inputs)[0]
                    loss = nn.functional.cross_entropy(logits, targets)
                    if training:
                        loss.backward()
                        optimizer.step()
                losses.append(float(loss.detach().item()))
            return sum(losses) / len(losses) if losses else 0.0

        train_losses = []
        validation_losses = []
        for epoch in range(self.epochs):
            order = list(range(len(split["train"])))
            random.Random(self.random_seed + epoch).shuffle(order)
            train_rows = [split["train"][index] for index in order]
            train_losses.append(loss_for(train_rows, training=True))
            validation_losses.append(loss_for(split["validation"]))
            if progress_callback:
                progress_callback(
                    f"epoch={epoch + 1}/{self.epochs} train_loss={train_losses[-1]:.6f} "
                    f"validation_loss={validation_losses[-1]:.6f}"
                )

        checkpoint = self._checkpoint_path("ner-local")
        torch.save(model.cpu().state_dict(), checkpoint / "weights.pt")
        metadata = {
            "task": "ner",
            "architecture": "local-context-feature-classifier",
            "feature_to_id": feature_to_id,
            "known_tokens": known_tokens,
            "unknown_token_policy": "outside",
            "id_to_label": {str(index): label for label, index in label_to_id.items()},
            "embedding_dim": embedding_dim,
            "seed": self.random_seed,
            "epochs": self.epochs,
            "learning_rate": float(self.config.get("learning_rate", 1e-2)),
            "split_sizes": {key: len(rows) for key, rows in split.items()},
            "split_indices": split_index_map,
            "split_source": "dataset" if any(example.metadata.get("split") for example in selected) else "seeded_random",
            "dataset_sha256": dataset_digest(selected),
            "training_examples": len(split["train"]),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "synthetic_data": self.synthetic_data,
            "device": str(device),
            "train_loss": train_losses,
            "validation_loss": validation_losses,
        }
        (checkpoint / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        evaluation = self.evaluate_ner(split["test"], checkpoint)
        baseline = self._ner_all_outside_baseline(split["test"])
        metadata["evaluation"] = evaluation
        metadata["baseline"] = baseline
        (checkpoint / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return {
            "status": "completed",
            "task": "ner",
            "checkpoint": str(checkpoint),
            "architecture": metadata["architecture"],
            "train_loss": train_losses,
            "validation_loss": validation_losses,
            "split_sizes": metadata["split_sizes"],
            "synthetic_data": self.synthetic_data,
            "evaluation": evaluation,
            "baseline": baseline,
            "interpretation": "Synthetic smoke-test data; these metrics are not meaningful model-performance estimates." if self.synthetic_data else None,
        }

    @staticmethod
    def evaluate_ner(dataset: Sequence[DatasetExample], checkpoint: str | Path) -> Dict[str, Any]:
        metadata = json.loads((Path(checkpoint) / "metadata.json").read_text(encoding="utf-8"))
        id_to_label = {int(key): value for key, value in metadata["id_to_label"].items()}
        architecture = metadata.get("architecture", "pretrained-transformer-token-classification")
        recognizer = None
        if architecture == "local-context-feature-classifier":
            recognizer = EntityRecognizer(checkpoint)
            tokenizer = None
            model = None
        else:
            try:
                from transformers import AutoModelForTokenClassification, AutoTokenizer
            except ImportError as error:
                raise RuntimeError("Transformer NER evaluation requires `transformers`.") from error
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
            model = AutoModelForTokenClassification.from_pretrained(checkpoint, local_files_only=True)
            model.eval()
        expected_entities = set()
        predicted_entities = set()
        for example_index, example in enumerate(dataset):
            if recognizer is not None:
                predicted_labels = recognizer.predict_tags(example.ner_tokens)
            else:
                encoded = tokenizer(example.ner_tokens, is_split_into_words=True, return_tensors="pt", truncation=True)
                word_ids = encoded.word_ids()
                with torch.no_grad():
                    logits = model(**encoded).logits[0]
                predicted_labels = ["O"] * len(example.ner_tokens)
                seen = set()
                for token_index, word_index in enumerate(word_ids):
                    if word_index is None or word_index in seen:
                        continue
                    seen.add(word_index)
                    predicted_labels[word_index] = id_to_label[int(logits[token_index].argmax().item())]
            offsets = [(index, index + 1) for index in range(len(example.ner_tokens))]
            for item in _bio_entities(example.ner_tags, example.ner_tokens, offsets):
                expected_entities.add((example_index, item["start"], item["end"], item["label"]))
            for item in _bio_entities(predicted_labels, example.ner_tokens, offsets):
                predicted_entities.add((example_index, item["start"], item["end"], item["label"]))
        labels = sorted({item[3] for item in expected_entities | predicted_entities})
        per_class = {}
        for label in labels:
            expected = {item for item in expected_entities if item[3] == label}
            predicted = {item for item in predicted_entities if item[3] == label}
            true_positive = len(expected & predicted)
            precision = true_positive / len(predicted) if predicted else 0.0
            recall = true_positive / len(expected) if expected else 0.0
            per_class[label] = {
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                "support": len(expected),
            }
        true_positive = len(expected_entities & predicted_entities)
        precision = true_positive / len(predicted_entities) if predicted_entities else 0.0
        recall = true_positive / len(expected_entities) if expected_entities else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "macro_f1": sum(scores["f1"] for scores in per_class.values()) / len(labels) if labels else 0.0,
            "per_class": per_class,
            "support": len(expected_entities),
        }

    @staticmethod
    def _ner_all_outside_baseline(dataset: Sequence[DatasetExample]) -> Dict[str, Any]:
        gold = set()
        for example_index, example in enumerate(dataset):
            offsets = [(index, index + 1) for index in range(len(example.ner_tokens))]
            for entity in _bio_entities(example.ner_tags, example.ner_tokens, offsets):
                gold.add((example_index, entity["start"], entity["end"], entity["label"]))
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "macro_f1": 0.0,
            "per_class": {
                label: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": count}
                for label, count in sorted(
                    (label, sum(entity[3] == label for entity in gold))
                    for label in {entity[3] for entity in gold}
                )
            },
            "support": len(gold),
            "baseline": "all tokens outside; predicts no entities",
        }

    @staticmethod
    def _ner_loss(model: Any, tokenizer: Any, examples: Sequence[DatasetExample], label_to_id: Dict[str, int]) -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for example in examples:
                device = next(model.parameters()).device
                encoded = tokenizer(
                    example.ner_tokens, is_split_into_words=True, return_tensors="pt", truncation=True
                ).to(device)
                word_ids = encoded.word_ids()
                labels, previous = [], None
                for word_id in word_ids:
                    labels.append(-100 if word_id is None or word_id == previous else label_to_id[example.ner_tags[word_id]])
                    previous = word_id
                loss = model(**encoded, labels=torch.tensor([labels], dtype=torch.long, device=device)).loss
                losses.append(float(loss.item()))
        return sum(losses) / len(losses) if losses else 0.0

    @staticmethod
    def load_relation_checkpoint(checkpoint: str | Path) -> RelationExtractor:
        return RelationExtractor(checkpoint)
