from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .graph import KnowledgeGraph


def _write_dataset(path: str | Path, dataset: Dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def prepare_conll2003(source: str | Path, destination: str | Path, split: str) -> Path:
    """Convert a locally obtained CoNLL-2003 IOB file into aligned word/tag rows."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("CoNLL split must be train, validation, or test.")
    examples: List[Dict[str, Any]] = []
    tokens: List[str] = []
    tags: List[str] = []

    def finish() -> None:
        if tokens:
            examples.append({
                "text": " ".join(tokens),
                "ner_tokens": list(tokens),
                "ner_tags": list(tags),
                "metadata": {"split": split},
            })
            tokens.clear()
            tags.clear()

    for line_number, raw_line in enumerate(Path(source).read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            finish()
            continue
        columns = line.split()
        if columns[0] == "-DOCSTART-":
            finish()
            continue
        if len(columns) < 2:
            raise ValueError(f"Malformed CoNLL row at {source}:{line_number}; expected token and tag columns.")
        token, tag = columns[0], columns[-1]
        if tag == "O":
            normalized_tag = "O"
        elif tag.startswith(("B-", "I-")):
            normalized_tag = tag
        else:
            raise ValueError(f"Unsupported CoNLL tag {tag!r} at {source}:{line_number}.")
        tokens.append(token)
        tags.append(normalized_tag)
    finish()
    if not examples:
        raise ValueError(f"No sentences were found in CoNLL input: {source}")
    return _write_dataset(destination, {
        "dataset": "CoNLL-2003",
        "dataset_split": split,
        "source_file": Path(source).name,
        "examples": examples,
    })


def combine_datasets(
    sources: Mapping[str, str | Path],
    destination: str | Path,
    dataset_name: str,
) -> Path:
    examples: List[Dict[str, Any]] = []
    for split, source in sources.items():
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unsupported dataset split: {split}")
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
        rows = payload.get("examples") if isinstance(payload, dict) else None
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Prepared {split} dataset has no examples: {source}")
        for row in rows:
            row = dict(row)
            metadata = dict(row.get("metadata") or {})
            metadata["split"] = split
            row["metadata"] = metadata
            examples.append(row)
    if not all(split in {row.get("metadata", {}).get("split") for row in examples} for split in sources):
        raise ValueError("One or more requested dataset splits are empty.")
    return _write_dataset(destination, {
        "dataset": dataset_name,
        "examples": examples,
    })


def _relation_name(relation_id: str, relation_map: Mapping[str, Any]) -> str:
    value = relation_map.get(relation_id)
    if isinstance(value, dict):
        name = value.get("title") or value.get("name")
    else:
        name = value
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Relation {relation_id!r} has no name in the supplied relation map.")
    return name.strip()


def prepare_docred(
    source: str | Path,
    destination: str | Path,
    relation_map_path: str | Path,
    split: str,
    negative_ratio: int = 1,
) -> Path:
    """Convert DocRED documents into one labeled entity-pair classification row per relation."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("DocRED split must be train, validation, or test.")
    if negative_ratio < 0:
        raise ValueError("DocRED negative_ratio cannot be negative.")
    relation_map_payload = json.loads(Path(relation_map_path).read_text(encoding="utf-8"))
    if not isinstance(relation_map_payload, dict):
        raise ValueError("DocRED relation map must be a JSON object keyed by relation ID.")
    raw_documents = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(raw_documents, list):
        raise ValueError("DocRED input must be a JSON array of documents.")
    examples: List[Dict[str, Any]] = []
    for document_index, document in enumerate(raw_documents):
        sentences = document.get("sents")
        vertices = document.get("vertexSet")
        labels = document.get("labels", [])
        if not isinstance(sentences, list) or not isinstance(vertices, list) or not isinstance(labels, list):
            raise ValueError(f"DocRED document {document_index} is missing sents, vertexSet, or labels.")
        token_strings: List[str] = []
        token_offsets: List[Tuple[int, int]] = []
        sentence_offsets: List[Tuple[int, int]] = []
        text_length = 0
        for sentence in sentences:
            start_index = len(token_strings)
            for token in sentence:
                token = str(token)
                if token_strings:
                    text_length += 1
                start = text_length
                token_strings.append(token)
                text_length += len(token)
                token_offsets.append((start, start + len(token)))
            sentence_offsets.append((start_index, len(token_offsets)))
        text = " ".join(token_strings)
        if not text:
            continue
        entity_mentions = []
        for entity_index, mentions in enumerate(vertices):
            valid = []
            for mention in mentions:
                sent_id = int(mention["sent_id"])
                start_token, end_token = map(int, mention["pos"])
                if sent_id < 0 or sent_id >= len(sentence_offsets):
                    continue
                global_start = sentence_offsets[sent_id][0] + start_token
                global_end = sentence_offsets[sent_id][0] + end_token
                if global_start < 0 or global_end > len(token_offsets) or global_start >= global_end:
                    continue
                valid.append({
                    "start": token_offsets[global_start][0],
                    "end": token_offsets[global_end - 1][1],
                    "name": str(mention.get("name", text[token_offsets[global_start][0]:token_offsets[global_end - 1][1]])),
                })
            entity_mentions.append(sorted(valid, key=lambda mention: (mention["start"], mention["end"])))
        positive_pairs = set()
        document_examples = []
        for relation in labels:
            head, tail = int(relation["h"]), int(relation["t"])
            if head >= len(entity_mentions) or tail >= len(entity_mentions) or not entity_mentions[head] or not entity_mentions[tail]:
                continue
            head_mention, tail_mention = entity_mentions[head][0], entity_mentions[tail][0]
            positive_pairs.add((head, tail))
            evidence_ids = sorted({int(index) for index in relation.get("evidence", []) if 0 <= int(index) < len(sentences)})
            evidence = [" ".join(str(token) for token in sentences[index]) for index in evidence_ids]
            document_examples.append({
                "text": text,
                "relations": [{
                    "subject": {"start": head_mention["start"], "end": head_mention["end"]},
                    "object": {"start": tail_mention["start"], "end": tail_mention["end"]},
                    "label": _relation_name(str(relation["r"]), relation_map_payload),
                    "document_id": str(document.get("title", document_index)),
                    "evidence_sentence_ids": evidence_ids,
                    "evidence": evidence,
                }],
                "metadata": {"split": split, "document_id": str(document.get("title", document_index))},
            })
        negative_pairs = [
            (head, tail)
            for head in range(len(entity_mentions))
            for tail in range(len(entity_mentions))
            if head != tail
            and (head, tail) not in positive_pairs
            and entity_mentions[head]
            and entity_mentions[tail]
            and (
                entity_mentions[head][0]["end"] <= entity_mentions[tail][0]["start"]
                or entity_mentions[tail][0]["end"] <= entity_mentions[head][0]["start"]
            )
        ]
        rng = random.Random(f"{split}:{document.get('title', document_index)}")
        requested_negatives = min(len(negative_pairs), max(1, len(document_examples) * negative_ratio)) if negative_ratio else 0
        for head, tail in rng.sample(negative_pairs, requested_negatives):
            head_mention, tail_mention = entity_mentions[head][0], entity_mentions[tail][0]
            document_examples.append({
                "text": text,
                "relations": [{
                    "subject": {"start": head_mention["start"], "end": head_mention["end"]},
                    "object": {"start": tail_mention["start"], "end": tail_mention["end"]},
                    "label": "NO_RELATION",
                    "document_id": str(document.get("title", document_index)),
                    "evidence_sentence_ids": [],
                    "evidence": [],
                }],
                "metadata": {"split": split, "document_id": str(document.get("title", document_index))},
            })
        examples.extend(document_examples)
    if not examples:
        raise ValueError(f"No labeled DocRED entity pairs were found in {source}.")
    return _write_dataset(destination, {
        "dataset": "DocRED",
        "dataset_split": split,
        "source_file": Path(source).name,
        "relation_map_file": Path(relation_map_path).name,
        "examples": examples,
    })


def prepare_fever(source: str | Path, destination: str | Path) -> Path:
    """Normalize FEVER claims and retain gold evidence references without inventing passages."""
    label_mapping = {
        "SUPPORTS": "SUPPORTED",
        "REFUTES": "REFUTED",
        "NOT ENOUGH INFO": "NOT ENOUGH INFORMATION",
        "NOT ENOUGH INFORMATION": "NOT ENOUGH INFORMATION",
    }
    examples = []
    for line_number, raw in enumerate(Path(source).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        claim = row.get("claim")
        label = label_mapping.get(str(row.get("label", "")).upper())
        if not isinstance(claim, str) or not claim.strip() or label is None:
            raise ValueError(f"FEVER row {line_number} requires a claim and recognized gold label.")
        evidence_groups = []
        for group in row.get("evidence") or []:
            references = []
            for item in group:
                if isinstance(item, list) and len(item) >= 4 and item[2] is not None:
                    references.append({"page": str(item[2]), "sentence_id": item[3]})
            if references:
                evidence_groups.append(references)
        examples.append({
            "claim": claim,
            "label": label,
            "evidence_groups": evidence_groups,
            "metadata": {"source_id": row.get("id"), "dataset": "FEVER"},
        })
    if not examples:
        raise ValueError(f"No FEVER rows were found in {source}.")
    return _write_dataset(destination, {
        "dataset": "FEVER",
        "examples": examples,
    })


def load_kg_benchmark(data_dir: str | Path) -> Tuple[KnowledgeGraph, Dict[str, List[Tuple[str, str, str]]]]:
    """Load the standard train/valid/test TSV files used by WN18RR and FB15k-237."""
    root = Path(data_dir)
    split_files = {"train": "train.txt", "validation": "valid.txt", "test": "test.txt"}
    splits: Dict[str, List[Tuple[str, str, str]]] = {}
    graph = KnowledgeGraph(root.name)
    for split, filename in split_files.items():
        source = root / filename
        if not source.is_file():
            raise FileNotFoundError(f"Expected benchmark split file: {source}")
        triples = []
        for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            fields = raw.rstrip("\n").split("\t")
            if len(fields) != 3 or any(not value for value in fields):
                raise ValueError(f"Malformed triple at {source}:{line_number}; expected head<TAB>relation<TAB>tail.")
            head, relation, tail = fields
            triples.append((head, relation, tail))
            graph.add_fact(
                head,
                relation,
                tail,
                source=f"{root.name}:{split}",
                provenance={"benchmark_split": split, "source_file": filename},
            )
        if not triples:
            raise ValueError(f"Benchmark split is empty: {source}")
        splits[split] = triples
    return graph, splits


def benchmark_index_splits(
    splits: Mapping[str, Sequence[Tuple[str, str, str]]],
    entity_to_id: Mapping[str, int],
    relation_to_id: Mapping[str, int],
) -> Dict[str, List[Tuple[int, int, int]]]:
    result: Dict[str, List[Tuple[int, int, int]]] = {}
    for name in ("train", "validation", "test"):
        triples = splits.get(name, ())
        result[name] = [
            (entity_to_id[head], relation_to_id[relation], entity_to_id[tail])
            for head, relation, tail in triples
        ]
    return result
