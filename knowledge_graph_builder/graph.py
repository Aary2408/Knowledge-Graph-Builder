from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class Entity:
    entity_id: str
    label: str
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"entity_id": self.entity_id, "label": self.label, "attributes": self.attributes}


@dataclass
class Relation:
    subject: str
    predicate: str
    obj: str
    source: str = "manual"
    confidence: float = 1.0
    relation_type: str = "observed"
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.obj,
            "source": self.source,
            "confidence": self.confidence,
            "relation_type": self.relation_type,
            "provenance": self.provenance,
        }


class KnowledgeGraph:
    """A lightweight knowledge graph that supports entities, named relations, lineage, and JSON persistence."""

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []

    def add_entity(self, entity_id: str, label: str, attributes: Optional[Dict[str, Any]] = None) -> Entity:
        existing = self.entities.get(entity_id)
        merged_attributes = dict(existing.attributes) if existing else {}
        if attributes:
            merged_attributes.update(attributes)
        entity = Entity(entity_id=entity_id, label=label, attributes=merged_attributes)
        self.entities[entity_id] = entity
        return entity

    def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        source: str = "manual",
        confidence: float = 1.0,
        relation_type: str = "observed",
        provenance: Optional[Dict[str, Any]] = None,
    ) -> Relation:
        relation = Relation(
            subject=subject,
            predicate=predicate,
            obj=obj,
            source=source,
            confidence=confidence,
            relation_type=relation_type,
            provenance=provenance or {},
        )
        self.relations.append(relation)
        if subject not in self.entities:
            self.add_entity(subject, subject)
        if obj not in self.entities:
            self.add_entity(obj, obj)
        return relation

    def add_relation(self, *args, **kwargs):
        return self.add_fact(*args, **kwargs)

    def find_entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)

    def query(self, subject: Optional[str] = None, predicate: Optional[str] = None, obj: Optional[str] = None) -> List[Relation]:
        rows: List[Relation] = []
        for relation in self.relations:
            if subject is not None and relation.subject != subject:
                continue
            if predicate is not None and relation.predicate != predicate:
                continue
            if obj is not None and relation.obj != obj:
                continue
            rows.append(relation)
        return rows

    def has_fact(self, subject: str, predicate: str, obj: str) -> bool:
        return any(
            relation.subject == subject and relation.predicate == predicate and relation.obj == obj
            for relation in self.relations
        )

    def export_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "entities": [entity.to_dict() for entity in self.entities.values()],
            "relations": [relation.to_dict() for relation in self.relations],
        }

    @classmethod
    def load_json(cls, path: str | Path) -> "KnowledgeGraph":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        graph = cls(name=payload.get("name", "loaded"))
        for entity in payload.get("entities", []):
            graph.add_entity(entity["entity_id"], entity.get("label", entity["entity_id"]), entity.get("attributes", {}))
        for relation in payload.get("relations", []):
            graph.add_fact(
                relation["subject"],
                relation["predicate"],
                relation["object"],
                source=relation.get("source", "manual"),
                confidence=float(relation.get("confidence", 1.0)),
                relation_type=relation.get("relation_type", "observed"),
                provenance=relation.get("provenance", {}),
            )
        return graph

    def save_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.export_json(), indent=2), encoding="utf-8")
        return target

    def __len__(self) -> int:
        return len(self.relations)

    def __iter__(self) -> Iterable[Relation]:
        return iter(self.relations)
