from __future__ import annotations

import json
import math
import random
import uuid
import time
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .graph import KnowledgeGraph

Triple = Tuple[int, int, int]


def ranking_metrics(ranks: Sequence[int], hits_at: Sequence[int] = (1, 3, 10)) -> Dict[str, float]:
    if any(rank < 1 for rank in ranks):
        raise ValueError("Ranks must be positive, one-based integers.")
    if not ranks:
        raise ValueError("At least one held-out rank is required to calculate link-prediction metrics.")
    result = {"mrr": sum(1 / rank for rank in ranks) / len(ranks)}
    for cutoff in hits_at:
        if cutoff < 1:
            raise ValueError("Hits@K cutoffs must be positive.")
        result[f"hits@{cutoff}"] = sum(rank <= cutoff for rank in ranks) / len(ranks)
    return result


def _dependencies() -> Tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch_geometric.nn import RGCNConv
    except ImportError as error:
        raise RuntimeError(
            "GNN training requires PyTorch and PyTorch Geometric. Install the CPU/GPU-compatible PyTorch build, "
            "then install the project dependencies from requirements.txt."
        ) from error
    return torch, nn, RGCNConv


def negative_sample(
    positives: Sequence[Triple],
    known_triples: set[Triple],
    num_entities: int,
    count: int,
    seed: int,
) -> List[Triple]:
    if count < 0:
        raise ValueError("Negative sample count cannot be negative.")
    if count == 0:
        return []
    if not positives:
        raise ValueError("At least one positive triple is required for negative sampling.")
    if num_entities < 2:
        raise ValueError("Negative sampling requires at least two entities.")
    rng = random.Random(seed)
    negatives: List[Triple] = []
    seen = set(known_triples)
    attempts = 0
    max_attempts = max(100, count * max(20, num_entities * 4))
    while len(negatives) < count and attempts < max_attempts:
        source, relation, target = positives[rng.randrange(len(positives))]
        corrupted = (source, relation, rng.randrange(num_entities))
        attempts += 1
        if corrupted not in seen and corrupted[0] != corrupted[2]:
            negatives.append(corrupted)
            seen.add(corrupted)
    if len(negatives) != count:
        raise ValueError(
            f"Could create only {len(negatives)} of {count} requested negatives; the graph may be too dense."
        )
    return negatives


def _make_model(num_entities: int, num_relations: int, hidden_dim: int = 64) -> Any:
    torch, nn, RGCNConv = _dependencies()

    class RGCNLinkPredictor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.entity_embedding = nn.Embedding(num_entities, hidden_dim)
            message_relation_count = max(1, num_relations * 2)
            self.conv1 = RGCNConv(hidden_dim, hidden_dim, message_relation_count, num_bases=min(16, message_relation_count))
            self.conv2 = RGCNConv(hidden_dim, hidden_dim, message_relation_count, num_bases=min(16, message_relation_count))
            self.relation_embedding = nn.Embedding(num_relations, hidden_dim)
            nn.init.xavier_uniform_(self.entity_embedding.weight)
            nn.init.xavier_uniform_(self.relation_embedding.weight)

        def encode(self, edge_index: Any, edge_type: Any) -> Any:
            nodes = torch.arange(num_entities, device=edge_index.device)
            hidden = torch.relu(self.conv1(self.entity_embedding(nodes), edge_index, edge_type))
            return self.conv2(hidden, edge_index, edge_type)

        def score(self, entity_features: Any, triples: Any) -> Any:
            source = entity_features[triples[:, 0]]
            relation = self.relation_embedding(triples[:, 1])
            target = entity_features[triples[:, 2]]
            return (source * relation * target).sum(dim=-1)

        def all_tail_scores(self, entity_features: Any, source: int, relation: int) -> Any:
            query = entity_features[source] * self.relation_embedding.weight[relation]
            return entity_features @ query

        def all_head_scores(self, entity_features: Any, relation: int, target: int) -> Any:
            query = self.relation_embedding.weight[relation] * entity_features[target]
            return entity_features @ query

    return RGCNLinkPredictor()


class GraphNeuralNetworkTrainer:
    """Relational graph convolution link predictor with train-only message passing."""

    def __init__(self, graph: Optional[KnowledgeGraph] = None, config: Optional[Dict[str, Any]] = None) -> None:
        self.graph = graph or KnowledgeGraph()
        self.config = config or {}
        self.random_seed = int(self.config.get("random_seed", 42))
        self.epochs = int(self.config.get("epochs", 100))
        self.learning_rate = float(self.config.get("learning_rate", 0.01))
        self.hidden_dim = int(self.config.get("hidden_dim", 64))
        self.device_preference = self.config.get("device", "auto")
        self.checkpoint_dir = Path(self.config.get("checkpoint_dir", "checkpoints/gnn"))
        if self.epochs < 1 or self.hidden_dim < 1:
            raise ValueError("epochs and hidden_dim must be positive integers.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be a finite positive number.")
        eligible = {"observed", "verified", "human_reviewed"}
        self.facts = [
            relation
            for relation in self.graph.relations
            if relation.relation_type in eligible
        ]
        self.entity_to_id = {entity_id: index for index, entity_id in enumerate(sorted(self.graph.entities))}
        self.relation_to_id = {name: index for index, name in enumerate(sorted({fact.predicate for fact in self.facts}))}
        self.triples: List[Triple] = [
            (self.entity_to_id[fact.subject], self.relation_to_id[fact.predicate], self.entity_to_id[fact.obj])
            for fact in self.facts
        ]
        self.known_triples = set(self.triples)
        self.model = None
        self.checkpoint: Optional[Path] = None
        self.split = self._split_triples()

    def _split_triples(self) -> Dict[str, List[Triple]]:
        benchmark_splits = self.config.get("benchmark_splits")
        if benchmark_splits is not None:
            if not isinstance(benchmark_splits, dict):
                raise ValueError("benchmark_splits must contain train, validation, and test triples.")
            split: Dict[str, List[Triple]] = {}
            for name in ("train", "validation", "test"):
                rows = benchmark_splits.get(name)
                if not isinstance(rows, list) or not rows:
                    raise ValueError(f"Benchmark {name} split must be a non-empty list of triples.")
                converted = []
                for triple in rows:
                    if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                        raise ValueError(f"Each benchmark {name} row must be a head/relation/tail triple.")
                    head, relation, tail = triple
                    try:
                        converted.append((self.entity_to_id[head], self.relation_to_id[relation], self.entity_to_id[tail]))
                    except KeyError as error:
                        raise ValueError(f"Benchmark split references an unknown graph mapping: {error.args[0]}") from error
                split[name] = converted
            flattened = [triple for rows in split.values() for triple in rows]
            if len(flattened) != len(set(flattened)):
                raise ValueError("Benchmark train/validation/test splits must not contain duplicate triples.")
            if set(flattened) != set(self.triples):
                raise ValueError("Benchmark splits must cover exactly the graph triples.")
            return split
        # Group both directions of an entity pair so a held-out edge cannot leak
        # through a reverse edge in the message-passing graph.
        pair_groups: Dict[Tuple[int, int], List[Triple]] = {}
        for triple in self.triples:
            pair = tuple(sorted((triple[0], triple[2])))
            pair_groups.setdefault(pair, []).append(triple)
        groups = list(pair_groups.values())
        random.Random(self.random_seed).shuffle(groups)
        if len(groups) < 3:
            return {"train": [], "validation": [], "test": [triple for group in groups for triple in group]}
        test_count = max(1, round(len(groups) * 0.2))
        validation_count = max(1, round(len(groups) * 0.1))
        if test_count + validation_count >= len(groups):
            test_count, validation_count = 1, 1
        test_groups = groups[:test_count]
        validation_groups = groups[test_count : test_count + validation_count]
        train_groups = groups[test_count + validation_count :]
        return {
            "train": [triple for group in train_groups for triple in group],
            "validation": [triple for group in validation_groups for triple in group],
            "test": [triple for group in test_groups for triple in group],
        }

    def prepare_dataset(self) -> Dict[str, Any]:
        return {
            "entity_to_id": dict(self.entity_to_id),
            "relation_to_id": dict(self.relation_to_id),
            "splits": {key: list(value) for key, value in self.split.items()},
            "split_source": "benchmark" if self.config.get("benchmark_splits") else "seeded_random",
            "message_passing_edges": list(self.split["train"]),
        }

    def _message_passing_tensors(self, triples: Sequence[Triple], device: Any) -> Tuple[Any, Any]:
        torch, _, _ = _dependencies()
        forward_edges = torch.tensor(
            [[source, target] for source, _, target in triples],
            dtype=torch.long,
            device=device,
        ).t().contiguous()
        forward_types = torch.tensor([relation for _, relation, _ in triples], dtype=torch.long, device=device)
        edge_index = torch.cat((forward_edges, forward_edges.flip(0)), dim=1)
        inverse_types = forward_types + len(self.relation_to_id)
        edge_type = torch.cat((forward_types, inverse_types))
        return edge_index, edge_type

    def train(self, progress_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        if len(self.split["train"]) < 1 or not self.split["validation"] or not self.split["test"]:
            return {
                "status": "insufficient_data",
                "warning": "At least three distinct undirected entity pairs are required for non-empty train, validation, and test splits.",
                "split_sizes": {key: len(rows) for key, rows in self.split.items()},
                "metrics": None,
                "checkpoint": None,
            }
        if len(self.facts) < 30:
            warning = "Small graph: metrics are smoke-test measurements and are not reliable performance estimates."
        else:
            warning = None
        torch, _, _ = _dependencies()
        started = time.perf_counter()
        if self.device_preference == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.device_preference)
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable; set device to `auto` or `cpu`.")
        torch.manual_seed(self.random_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(self.random_seed)
        random.seed(self.random_seed)
        triples = torch.tensor(self.split["train"], dtype=torch.long, device=device)
        edge_index, edge_type = self._message_passing_tensors(self.split["train"], device)
        model = _make_model(len(self.entity_to_id), len(self.relation_to_id), self.hidden_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        train_losses, validation_losses = [], []
        for epoch in range(self.epochs):
            model.train()
            optimizer.zero_grad()
            entity_features = model.encode(edge_index, edge_type)
            negatives = negative_sample(
                self.split["train"], self.known_triples, len(self.entity_to_id),
                len(self.split["train"]), self.random_seed + epoch,
            )
            negative_tensor = torch.tensor(negatives, dtype=torch.long, device=device)
            positive_scores = model.score(entity_features, triples)
            negative_scores = model.score(entity_features, negative_tensor)
            loss = torch.nn.functional.softplus(-positive_scores).mean() + torch.nn.functional.softplus(negative_scores).mean()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().item()))
            model.eval()
            with torch.no_grad():
                features = model.encode(edge_index, edge_type)
                val = torch.tensor(self.split["validation"], dtype=torch.long, device=device)
                val_negatives = negative_sample(
                    self.split["validation"], self.known_triples, len(self.entity_to_id),
                    len(self.split["validation"]), self.random_seed + 10000 + epoch,
                )
                val_negative_tensor = torch.tensor(val_negatives, dtype=torch.long, device=device)
                val_loss = torch.nn.functional.softplus(-model.score(features, val)).mean()
                val_loss += torch.nn.functional.softplus(model.score(features, val_negative_tensor)).mean()
                validation_losses.append(float(val_loss.item()))
            if progress_callback:
                progress_callback(
                    f"epoch={epoch + 1}/{self.epochs} train_loss={train_losses[-1]:.6f} "
                    f"validation_loss={validation_losses[-1]:.6f}"
                )

        model.eval()
        with torch.no_grad():
            entity_features = model.encode(edge_index, edge_type)
            model_ranks = self._ranks(model, entity_features, self.split["test"])
            baseline_ranks = self._degree_baseline_ranks(self.split["test"])
        metrics = ranking_metrics(model_ranks)
        baseline = ranking_metrics(baseline_ranks)
        runtime_seconds = time.perf_counter() - started
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = self.checkpoint_dir / f"rgcn-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "entity_to_id": self.entity_to_id,
                "relation_to_id": self.relation_to_id,
                "config": {
                    "hidden_dim": self.hidden_dim,
                    "random_seed": self.random_seed,
                    "epochs": self.epochs,
                    "learning_rate": self.learning_rate,
                    "device": self.device_preference,
                    "actual_device": str(device),
                    "benchmark_splits": self.config.get("benchmark_splits"),
                    "dataset_name": self.config.get("dataset_name", self.graph.name),
                    "inverse_relation_to_id": {
                        f"inverse::{name}": index + len(self.relation_to_id)
                        for name, index in self.relation_to_id.items()
                    },
                },
                "metrics": metrics,
                "baseline": baseline,
                "training_edges": self.split["train"],
                "splits": self.split,
                "known_triples": list(self.known_triples),
                "training_history": {
                    "train_loss": train_losses,
                    "validation_loss": validation_losses,
                    "small_graph_warning": warning,
                    "device": str(device),
                    "trained_at": datetime.now(timezone.utc).isoformat(),
                    "architecture": "rgcn-distmult-link-predictor",
                    "dataset_size": {
                        "entities": len(self.entity_to_id),
                        "relations": len(self.relation_to_id),
                        "triples": len(self.triples),
                    },
                    "runtime_seconds": runtime_seconds,
                    "query_count": len(model_ranks),
                    "evaluation_protocol": "filtered tail and head prediction over all known benchmark triples",
                },
            },
            checkpoint,
        )
        self.model = model
        self.checkpoint = checkpoint
        return {
            "status": "completed",
            "task": "rgcn_link_prediction",
            "checkpoint": str(checkpoint),
            "train_loss": train_losses,
            "validation_loss": validation_losses,
            "metrics": metrics,
            "baseline": baseline,
            "split_sizes": {key: len(rows) for key, rows in self.split.items()},
            "warning": warning,
            "device": str(device),
            "runtime_seconds": runtime_seconds,
            "dataset_size": {
                "entities": len(self.entity_to_id),
                "relations": len(self.relation_to_id),
                "triples": len(self.triples),
            },
            "query_count": len(model_ranks),
            "evaluation_protocol": "filtered tail and head prediction over all known benchmark triples",
        }

    def _ranks(self, model: Any, features: Any, positives: Sequence[Triple]) -> List[int]:
        torch, _, _ = _dependencies()
        ranks: List[int] = []
        device = features.device
        relation_embeddings = model.relation_embedding.weight
        for offset in range(0, len(positives), 128):
            batch = positives[offset : offset + 128]
            sources = torch.tensor([triple[0] for triple in batch], dtype=torch.long, device=device)
            relations = torch.tensor([triple[1] for triple in batch], dtype=torch.long, device=device)
            targets = torch.tensor([triple[2] for triple in batch], dtype=torch.long, device=device)
            relation_features = relation_embeddings[relations]
            tail_scores = ((features[sources] * relation_features) @ features.t()).detach().cpu()
            head_scores = ((features[targets] * relation_features) @ features.t()).detach().cpu()
            for index, (source, relation, target) in enumerate(batch):
                tail_row = tail_scores[index]
                target_score = tail_row[target]
                greater = int((tail_row > target_score).sum().item())
                filtered_tails = {
                    other_target
                    for other_source, other_relation, other_target in self.known_triples
                    if other_source == source and other_relation == relation and other_target != target
                }
                greater -= sum(bool(tail_row[candidate] > target_score) for candidate in filtered_tails)
                ranks.append(1 + greater)

                head_row = head_scores[index]
                source_score = head_row[source]
                greater = int((head_row > source_score).sum().item())
                filtered_heads = {
                    other_source
                    for other_source, other_relation, other_target in self.known_triples
                    if other_relation == relation and other_target == target and other_source != source
                }
                greater -= sum(bool(head_row[candidate] > source_score) for candidate in filtered_heads)
                ranks.append(1 + greater)
        return ranks

    def _degree_baseline_ranks(self, positives: Sequence[Triple]) -> List[int]:
        degree = {entity: 0 for entity in range(len(self.entity_to_id))}
        for source, _, target in self.split["train"]:
            degree[source] += 1
            degree[target] += 1
        sorted_degrees = sorted(degree.values())
        ranks = []
        for source, relation, target in positives:
            greater = len(sorted_degrees) - bisect_right(sorted_degrees, degree[target])
            filtered_tails = {
                other_target
                for other_source, other_relation, other_target in self.known_triples
                if other_source == source and other_relation == relation and other_target != target
            }
            greater -= sum(degree[candidate] > degree[target] for candidate in filtered_tails)
            ranks.append(1 + greater)
            filtered_heads = {
                other_source
                for other_source, other_relation, other_target in self.known_triples
                if other_relation == relation and other_target == target and other_source != source
            }
            greater = len(sorted_degrees) - bisect_right(sorted_degrees, degree[source])
            greater -= sum(degree[candidate] > degree[source] for candidate in filtered_heads)
            ranks.append(1 + greater)
        return ranks

    @classmethod
    def load_checkpoint(cls, checkpoint: str | Path, graph: KnowledgeGraph) -> "GraphNeuralNetworkTrainer":
        torch, _, _ = _dependencies()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        trainer = cls(graph, {**saved["config"], "checkpoint_dir": str(Path(checkpoint).parent)})
        if trainer.entity_to_id != saved["entity_to_id"] or trainer.relation_to_id != saved["relation_to_id"]:
            raise ValueError("The provided graph mappings do not match the GNN checkpoint.")
        if trainer.known_triples != set(saved["known_triples"]):
            raise ValueError("The provided graph facts do not match the GNN checkpoint's training graph.")
        trainer.split = saved["splits"]
        trainer.known_triples = set(saved["known_triples"])
        if trainer.device_preference == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(trainer.device_preference)
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable; set device to `auto` or `cpu`.")
        trainer.model = _make_model(
            len(trainer.entity_to_id), len(trainer.relation_to_id), saved["config"]["hidden_dim"]
        ).to(device)
        trainer.model.load_state_dict(saved["state_dict"])
        trainer.model.eval()
        trainer.checkpoint = Path(checkpoint)
        return trainer

    def evaluate_checkpoint(self) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Load a trained GNN checkpoint before evaluation.")
        if not self.split["test"]:
            raise ValueError("Checkpoint does not contain a valid held-out test split.")
        torch, _, _ = _dependencies()
        device = next(self.model.parameters()).device
        edge_index, edge_type = self._message_passing_tensors(self.split["train"], device)
        with torch.no_grad():
            features = self.model.encode(edge_index, edge_type)
            model_ranks = self._ranks(self.model, features, self.split["test"])
            baseline_ranks = self._degree_baseline_ranks(self.split["test"])
        return {
            "metrics": ranking_metrics(model_ranks),
            "baseline": ranking_metrics(baseline_ranks),
            "query_count": len(model_ranks),
            "evaluation_protocol": "filtered tail and head prediction over all known benchmark triples",
        }

    def predict(self, source_entity: str, predicate: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("Load a trained GNN checkpoint before inference.")
        if source_entity not in self.entity_to_id or predicate not in self.relation_to_id:
            raise ValueError("Source entity and relation must exist in the trained checkpoint mappings.")
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        torch, _, _ = _dependencies()
        device = next(self.model.parameters()).device
        edge_index, edge_type = self._message_passing_tensors(self.split["train"], device)
        with torch.no_grad():
            features = self.model.encode(edge_index, edge_type)
            scores = self.model.all_tail_scores(features, self.entity_to_id[source_entity], self.relation_to_id[predicate])
        known = {
            target for source, rel, target in self.known_triples
            if source == self.entity_to_id[source_entity] and rel == self.relation_to_id[predicate]
        }
        candidates = [
            (entity, float(scores[index].item()))
            for entity, index in self.entity_to_id.items()
            if index != self.entity_to_id[source_entity] and index not in known
        ]
        candidates.sort(key=lambda item: item[1], reverse=True)
        return [
            {
                "subject": source_entity,
                "predicate": predicate,
                "object": entity,
                "score": score,
                "score_type": "raw_distmult_logit",
                "label": "predicted_candidate",
            }
            for entity, score in candidates[:top_k]
        ]
