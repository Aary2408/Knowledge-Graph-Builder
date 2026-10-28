from __future__ import annotations

import re
import json
import math
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .graph import KnowledgeGraph, Relation
from .nlp import classification_metrics, dataset_digest, split_indices

VERIFICATION_LABELS = ("SUPPORTED", "REFUTED", "NOT ENOUGH INFORMATION")


@dataclass
class EvidenceRecord:
    source: str
    snippet: str
    score: float
    graph_path: List[str]
    provenance: Dict[str, Any] = field(default_factory=dict)


def _contains(text: str, phrase: str) -> bool:
    words = [part for part in re.split(r"[_\W]+", phrase.lower()) if part]
    if not words:
        return False
    pattern = r"(?<!\w)" + r"[\W_]+".join(re.escape(word) for word in words) + r"(?!\w)"
    return re.search(pattern, text.lower()) is not None


def _negated_fact(claim: str, relation: Relation) -> bool:
    subject = re.escape(relation.subject.lower())
    predicate_parts = relation.predicate.lower().split("_")
    predicate = r"[\W_]+".join(
        re.escape(part[:-1]) + "s?" if part.endswith("s") else re.escape(part)
        for part in predicate_parts
    )
    obj = re.escape(relation.obj.lower())
    claim = claim.lower()
    pattern = rf"(?<!\w){subject}(?!\w).{{0,40}}\b(?:does not|do not|did not|is not|was not|not)\b.{{0,30}}{predicate}.{{0,40}}(?<!\w){obj}(?!\w)"
    return re.search(pattern, claim) is not None


class TransformerNLIClassifier:
    """Local, fine-tuned transformer classifier for premise/hypothesis pairs."""

    def __init__(self, checkpoint: str | Path) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Transformer NLI requires PyTorch and Transformers.") from error
        self.torch = torch
        self.checkpoint = Path(checkpoint)
        metadata = json.loads((self.checkpoint / "metadata.json").read_text(encoding="utf-8"))
        self.id_to_label = {int(key): value for key, value in metadata["id_to_label"].items()}
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint, local_files_only=True)
        self.model.eval()

    def predict(self, premise: str, hypothesis: str) -> Dict[str, Any]:
        encoded = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
        with self.torch.no_grad():
            logits = self.model(**encoded).logits[0]
            probabilities = self.torch.softmax(logits, dim=-1)
            label_id = int(logits.argmax().item())
        return {
            "label": self.id_to_label[label_id],
            "confidence": float(probabilities[label_id].item()),
            "confidence_type": "uncalibrated_softmax_probability",
        }


class ClaimNLITrainer:
    """Fine-tunes a pretrained sequence-classification transformer on labeled NLI rows."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.dataset_path = self.config.get("dataset_path")
        self.checkpoint_dir = Path(self.config.get("checkpoint_dir", "checkpoints/verification"))
        self.pretrained_model = self.config.get("pretrained_model", "cross-encoder/nli-distilroberta-base")
        self.epochs = int(self.config.get("epochs", 3))
        self.batch_size = int(self.config.get("batch_size", 8))
        self.learning_rate = float(self.config.get("learning_rate", 2e-5))
        self.seed = int(self.config.get("random_seed", 42))
        self.device_preference = self.config.get("device", "auto")
        self.synthetic_data = bool(self.config.get("synthetic_data", False))
        if self.epochs < 1 or self.batch_size < 1 or not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("epochs, batch_size, and learning_rate must be positive.")

    def load_dataset(self, path: Optional[str | Path] = None) -> List[Dict[str, str]]:
        source_value = path or self.dataset_path
        if not source_value:
            raise FileNotFoundError("Provide `dataset_path`; no claim-verification dataset is configured.")
        source = Path(source_value)
        if not source.exists():
            raise FileNotFoundError(f"Claim-verification dataset was not found: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, (dict, list)):
            raise ValueError("NLI dataset must be a JSON object or array.")
        rows = payload if isinstance(payload, list) else payload.get("examples")
        self.synthetic_data = bool(payload.get("synthetic", False)) if isinstance(payload, dict) else False
        if not isinstance(rows, list) or not rows:
            raise ValueError("NLI dataset must contain a non-empty `examples` array.")
        normalized = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"NLI example {index} must be a JSON object.")
            premise, hypothesis, label = row.get("premise"), row.get("hypothesis"), row.get("label")
            if not all(isinstance(value, str) and value.strip() for value in (premise, hypothesis, label)):
                raise ValueError(f"NLI example {index} requires non-empty premise, hypothesis, and label strings.")
            if label not in VERIFICATION_LABELS:
                raise ValueError(f"NLI label must be one of {', '.join(VERIFICATION_LABELS)}.")
            normalized.append({"premise": premise, "hypothesis": hypothesis, "label": label})
            split = row.get("split", (row.get("metadata") or {}).get("split"))
            if split is not None:
                normalized[-1]["split"] = split
        splits = [row.get("split") for row in normalized if "split" in row]
        if splits and (len(splits) != len(normalized) or set(splits) != {"train", "validation", "test"}):
            raise ValueError("Explicit NLI splits must label every row and include train, validation, and test.")
        if len(normalized) < 3:
            raise ValueError("At least 3 NLI examples are required for train/validation/test splits.")
        return normalized

    def _split(self, rows: Sequence[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
        explicit_splits = [row.get("split") for row in rows if "split" in row]
        if explicit_splits:
            if len(explicit_splits) != len(rows) or set(explicit_splits) != {"train", "validation", "test"}:
                raise ValueError("Explicit NLI splits must label every row and include train, validation, and test.")
            fixed = {
                name: [row for row in rows if row.get("split") == name]
                for name in ("train", "validation", "test")
            }
            if not all(fixed.values()):
                raise ValueError("Each explicit NLI split must contain examples.")
            return fixed
        indices = split_indices(len(rows), self.seed)
        return {key: [rows[index] for index in values] for key, values in indices.items()}

    def train(
        self,
        rows: Optional[Sequence[Dict[str, str]]] = None,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Claim NLI training requires PyTorch and Transformers.") from error
        examples = list(rows) if rows is not None else self.load_dataset()
        if not examples:
            raise ValueError("NLI dataset is empty.")
        split = self._split(examples)
        if examples and all(row.get("split") in {"train", "validation", "test"} for row in examples):
            split_index_map = {
                name: [index for index, row in enumerate(examples) if row.get("split") == name]
                for name in ("train", "validation", "test")
            }
        else:
            split_index_map = split_indices(len(examples), self.seed)
        unseen = set(VERIFICATION_LABELS) - {row["label"] for row in split["train"]}
        if unseen:
            raise ValueError(f"Training split lacks verification labels: {sorted(unseen)}.")
        label_to_id = {label: index for index, label in enumerate(VERIFICATION_LABELS)}
        id_to_label = {index: label for label, index in label_to_id.items()}
        torch.manual_seed(self.seed)
        if self.device_preference == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.device_preference)
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable; set device to `auto` or `cpu`.")
        if device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.pretrained_model,
                num_labels=len(label_to_id),
                id2label=id_to_label,
                label2id=label_to_id,
                ignore_mismatched_sizes=True,
            ).to(device)
        except Exception as error:
            raise RuntimeError(
                f"Could not load pretrained NLI model `{self.pretrained_model}`. Check network access/cache or set `pretrained_model` to a local Hugging Face model directory."
            ) from error
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        train_losses, validation_losses = [], []
        for epoch in range(self.epochs):
            model.train()
            indices = list(range(len(split["train"])))
            random.Random(self.seed + epoch).shuffle(indices)
            epoch_losses = []
            for offset in range(0, len(indices), self.batch_size):
                batch = [split["train"][index] for index in indices[offset : offset + self.batch_size]]
                encoded = tokenizer(
                    [row["premise"] for row in batch],
                    [row["hypothesis"] for row in batch],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                labels = torch.tensor([label_to_id[row["label"]] for row in batch], dtype=torch.long, device=device)
                optimizer.zero_grad()
                loss = model(**encoded, labels=labels).loss
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().item()))
            train_losses.append(sum(epoch_losses) / len(epoch_losses))
            validation_losses.append(self._loss(model, tokenizer, split["validation"], label_to_id, device))
            if progress_callback:
                progress_callback(
                    f"epoch={epoch + 1}/{self.epochs} train_loss={train_losses[-1]:.6f} "
                    f"validation_loss={validation_losses[-1]:.6f}"
                )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = self.checkpoint_dir / f"claim-nli-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        checkpoint.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        metadata = {
            "task": "claim_nli",
            "architecture": "pretrained-transformer-sequence-classification",
            "id_to_label": {str(key): value for key, value in id_to_label.items()},
            "pretrained_model": self.pretrained_model,
            "seed": self.seed,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "split_sizes": {key: len(value) for key, value in split.items()},
            "split_indices": split_index_map,
            "dataset_sha256": dataset_digest(examples),
            "device": str(device),
            "synthetic_data": self.synthetic_data,
        }
        (checkpoint / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        evaluation = self.evaluate_checkpoint(checkpoint, split["test"])
        majority_label = max(
            VERIFICATION_LABELS,
            key=lambda label: sum(row["label"] == label for row in split["train"]),
        )
        baseline_expected = [row["label"] for row in split["test"]]
        baseline_predicted = [majority_label] * len(split["test"])
        baseline = classification_metrics(baseline_expected, baseline_predicted)
        metadata["train_loss"] = train_losses
        metadata["validation_loss"] = validation_losses
        metadata["evaluation"] = evaluation
        metadata["baseline"] = {
            "name": "training-split majority class",
            "label": majority_label,
            "evaluation": baseline,
        }
        (checkpoint / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return {
            "status": "completed",
            "task": "claim_nli",
            "checkpoint": str(checkpoint),
            "train_loss": train_losses,
            "validation_loss": validation_losses,
            "split_sizes": metadata["split_sizes"],
            "evaluation": evaluation,
            "baseline": metadata["baseline"],
            "synthetic_data": self.synthetic_data,
            "interpretation": "Synthetic smoke-test data; these metrics are not meaningful model-performance estimates." if self.synthetic_data else None,
        }

    @staticmethod
    def _loss(model: Any, tokenizer: Any, rows: Sequence[Dict[str, str]], label_to_id: Dict[str, int], device: Any) -> float:
        import torch

        model.eval()
        losses = []
        with torch.no_grad():
            for row in rows:
                encoded = tokenizer(row["premise"], row["hypothesis"], return_tensors="pt", truncation=True).to(device)
                labels = torch.tensor([label_to_id[row["label"]]], dtype=torch.long, device=device)
                losses.append(float(model(**encoded, labels=labels).loss.item()))
        return sum(losses) / len(losses) if losses else 0.0

    @staticmethod
    def evaluate_checkpoint(checkpoint: str | Path, rows: Sequence[Dict[str, str]]) -> Dict[str, Any]:
        classifier = TransformerNLIClassifier(checkpoint)
        expected = [row["label"] for row in rows]
        predicted = [classifier.predict(row["premise"], row["hypothesis"])["label"] for row in rows]
        result = classification_metrics(expected, predicted)
        labels = sorted(set(expected) | set(predicted))
        result["confusion_matrix"] = {
            actual: {
                guess: sum(label == actual and prediction == guess for label, prediction in zip(expected, predicted))
                for guess in labels
            }
            for actual in labels
        }
        return result


class ClaimVerifier:
    """Conservative graph-evidence verifier; absent facts never imply refutation."""

    def __init__(self, graph: Optional[KnowledgeGraph] = None, config: Optional[Dict[str, Any]] = None) -> None:
        self.graph = graph or KnowledgeGraph()
        self.config = config or {}
        checkpoint = self.config.get("nli_checkpoint")
        self.nli_classifier = TransformerNLIClassifier(checkpoint) if checkpoint else None

    def retrieve_evidence(self, claim: str) -> List[EvidenceRecord]:
        evidence = []
        for relation in self.graph.relations:
            if relation.relation_type not in {"observed", "verified", "human_reviewed"}:
                continue
            if any(_contains(claim, value) for value in (relation.subject, relation.predicate, relation.obj)):
                evidence.append(
                    EvidenceRecord(
                        source=relation.source,
                        snippet=relation.provenance.get(
                            "snippet", f"{relation.subject} {relation.predicate} {relation.obj}"
                        ),
                        score=float(relation.confidence),
                        graph_path=[relation.subject, relation.predicate, relation.obj],
                        provenance=dict(relation.provenance),
                    )
                )
        return evidence

    def verify(self, claim: str) -> Dict[str, Any]:
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("Claim must be a non-empty string.")
        evidence = self.retrieve_evidence(claim)
        classifier_confidence = None
        conflict = False
        evidence_predictions: List[Dict[str, Any]] = []
        if self.nli_classifier and evidence:
            results = [
                self.nli_classifier.predict(record.snippet, claim)
                for record in evidence
            ]
            evidence_predictions = results
            supported = [result for result in results if result["label"] == "SUPPORTED"]
            refuted = [result for result in results if result["label"] == "REFUTED"]
            conflict = bool(supported and refuted)
            if conflict:
                label = "NOT ENOUGH INFORMATION"
                classifier_confidence = max(item["confidence"] for item in results)
            else:
                winner = max(supported or refuted or results, key=lambda item: item["confidence"])
                label = winner["label"]
                classifier_confidence = winner["confidence"]
        else:
            supported = any(
                _contains(claim, relation.subject)
                and _contains(claim, relation.predicate)
                and _contains(claim, relation.obj)
                for relation in self.graph.relations
                if relation.relation_type in {"observed", "verified", "human_reviewed"}
            )
            refuted = not supported and any(
                _negated_fact(claim, relation)
                for relation in self.graph.relations
                if relation.relation_type in {"observed", "verified", "human_reviewed"}
            )
            label = "SUPPORTED" if supported else "REFUTED" if refuted else "NOT ENOUGH INFORMATION"
        evidence_strength = sum(item.score for item in evidence) / len(evidence) if evidence else 0.0
        evidence_output = []
        for index, record in enumerate(evidence):
            output = dict(record.__dict__)
            if evidence_predictions:
                output["classification"] = evidence_predictions[index]["label"]
                output["classifier_confidence"] = evidence_predictions[index]["confidence"]
            evidence_output.append(output)
        return {
            "label": label,
            "evidence_strength": evidence_strength,
            "classifier_confidence": classifier_confidence,
            "classifier_confidence_type": "uncalibrated_softmax_probability" if classifier_confidence is not None else None,
            "conflicting_evidence": conflict,
            "evidence": evidence_output,
            "model_version": (
                str(self.nli_classifier.checkpoint)
                if self.nli_classifier
                else self.config.get("model_version", "graph-exact-match-v1")
            ),
            "limitations": (
                "Predictions use a fine-tuned transformer over retrieved graph evidence."
                if self.nli_classifier
                else "Deterministic graph matching only; this is not a pretrained or fine-tuned NLI model."
            ),
        }

    def evaluate(self, claims: Sequence[str], labels: Sequence[str]) -> Dict[str, Any]:
        if len(claims) != len(labels):
            raise ValueError("claims and labels must be the same length")
        predicted = [self.verify(claim)["label"] for claim in claims]
        result = classification_metrics(labels, predicted)
        labels_used = sorted(set(labels) | set(predicted))
        result["confusion_matrix"] = {
            actual: {guess: sum(a == actual and p == guess for a, p in zip(labels, predicted)) for guess in labels_used}
            for actual in labels_used
        }
        return result

    def evaluate_fever(self, examples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        claims = [row["claim"] for row in examples]
        labels = [row["label"] for row in examples]
        classification = self.evaluate(claims, labels)
        normalize = lambda value: re.sub(r"\s+", " ", re.sub(r"[_\W]+", " ", str(value).lower())).strip()
        true_positive = retrieved_count = gold_count = 0
        evaluated_claims = 0
        nei_with_retrieval = 0
        nei_claims = 0
        for row in examples:
            retrieved = self.retrieve_evidence(row["claim"])
            retrieved_pages = {
                normalize(item.provenance.get("document_id", item.source))
                for item in retrieved
                if item.provenance.get("document_id", item.source)
            }
            if row["label"] == "NOT ENOUGH INFORMATION":
                nei_claims += 1
                nei_with_retrieval += bool(retrieved_pages)
                continue
            gold_pages = {
                normalize(reference["page"])
                for group in row.get("evidence_groups", [])
                for reference in group
                if reference.get("page")
            }
            if not gold_pages:
                continue
            evaluated_claims += 1
            true_positive += len(retrieved_pages & gold_pages)
            retrieved_count += len(retrieved_pages)
            gold_count += len(gold_pages)
        precision = true_positive / retrieved_count if retrieved_count else 0.0
        recall = true_positive / gold_count if gold_count else 0.0
        evidence_retrieval = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "evaluated_claims": evaluated_claims,
            "matching_basis": "normalized source/document IDs compared with FEVER gold evidence page IDs",
        }
        return {
            "classification": classification,
            "evidence_retrieval": evidence_retrieval,
            "nei_retrieval_rate": nei_with_retrieval / nei_claims if nei_claims else None,
            "nei_claims": nei_claims,
        }
