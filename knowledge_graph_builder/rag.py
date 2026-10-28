from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .graph import KnowledgeGraph, Relation
from .document_analysis import answer_from_document


@dataclass
class RetrievalResult:
    score: float
    entity: str
    snippet: str
    provenance: Dict[str, Any] = field(default_factory=dict)
    relation_type: str = "observed"
    graph_path: List[str] = field(default_factory=list)


class GraphRAGEngine:
    """Lexical graph retrieval that preserves provenance and separates predictions."""

    def __init__(self, graph: Optional[KnowledgeGraph] = None, config: Optional[Dict[str, Any]] = None) -> None:
        self.graph = graph or KnowledgeGraph()
        self.config = config or {}

    @staticmethod
    def _terms(value: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9]+", value.lower()) if len(token) > 1}

    def retrieve(self, query: str, top_k: int = 5, include_predictions: bool = False) -> List[RetrievalResult]:
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        query_terms = self._terms(query)
        eligible = [
            relation
            for relation in self.graph.relations
            if include_predictions
            or relation.relation_type not in {"predicted", "candidate", "model_prediction"}
        ]
        documents = [
            (relation, self._terms(f"{relation.subject} {relation.predicate} {relation.obj}"))
            for relation in eligible
        ]
        document_frequency = {
            term: sum(term in terms for _, terms in documents)
            for term in query_terms
        }
        average_length = sum(len(terms) for _, terms in documents) / max(1, len(documents))
        results: List[RetrievalResult] = []
        total_documents = len(documents)
        for relation, fact_terms in documents:
            is_prediction = relation.relation_type in {"predicted", "candidate", "model_prediction"}
            if is_prediction and not include_predictions:
                continue
            score = 0.0
            for term in query_terms & fact_terms:
                frequency = 1
                inverse_document_frequency = math.log(
                    1 + (total_documents - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5)
                )
                denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * len(fact_terms) / max(1, average_length))
                score += inverse_document_frequency * frequency * 2.5 / denominator
            if score > 0:
                results.append(
                    RetrievalResult(
                        score=score,
                        entity=relation.subject,
                        snippet=f"{relation.subject} {relation.predicate} {relation.obj}",
                        provenance={
                            "source": relation.source,
                            "confidence": relation.confidence,
                            **relation.provenance,
                        },
                        relation_type=relation.relation_type,
                        graph_path=[relation.subject, relation.predicate, relation.obj],
                    )
                )
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]

    def answer(self, query: str, document_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if document_context:
            answer = answer_from_document(
                query,
                document_context.get("text", ""),
                document_context.get("filename", document_context.get("document_id", "document")),
                document_context.get("document_id", ""),
                document_context.get("pages", []),
                document_context.get("candidate_relations", []),
            )
            supporting_graph = self.retrieve(query, top_k=int(self.config.get("top_k", 5)))
            graph_evidence = [
                {
                    "fact": match.snippet,
                    "source": match.provenance.get("source"),
                    "provenance": match.provenance,
                    "graph_path": match.graph_path,
                    "score": match.score,
                    "relation_type": match.relation_type,
                }
                for match in supporting_graph
                if match.provenance.get("document_id") == document_context.get("document_id")
            ]
            return {
                **answer,
                "evidence": [
                    {
                        "fact": item["excerpt"],
                        "excerpt": item["excerpt"],
                        "source": item["source"],
                        "document_id": item["document_id"],
                        "page": item["page"],
                        "source_offsets": item["source_offsets"],
                        "relation_type": "source_excerpt",
                    }
                    for item in answer["evidence"]
                ] + graph_evidence,
                "graph_evidence": graph_evidence,
                "limitations": "Answers retrieve matching excerpts from the selected source and supporting stored graph facts. Retrieval does not establish truth or generate independent citations.",
            }
        matches = self.retrieve(query, top_k=int(self.config.get("top_k", 5)))
        candidates = self.retrieve(query, top_k=int(self.config.get("top_k", 5)), include_predictions=True)
        candidates = [
            match
            for match in candidates
            if match.relation_type in {"predicted", "candidate", "model_prediction"} and match.score >= 0.5
        ]
        evidence = [
            {
                "fact": match.snippet,
                "source": match.provenance.get("source"),
                "provenance": match.provenance,
                "graph_path": match.graph_path,
                "score": match.score,
                "relation_type": match.relation_type,
            }
            for match in matches
        ]
        if not evidence:
            response = "I could not find supporting source evidence in the graph."
        else:
            response = "Retrieved graph evidence: " + "; ".join(item["fact"] for item in evidence)
        return {
            "answer": response,
            "evidence": evidence,
            "entities": sorted({match.entity for match in matches}),
            "predicted_candidates": [
                {"fact": match.snippet, "score": match.score, "label": "predicted_candidate"}
                for match in candidates
            ],
            "limitations": "Retrieval summarizes stored graph facts; it does not establish truth or generate independent citations.",
        }
