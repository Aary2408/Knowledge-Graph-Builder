from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_SENTENCE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")
_NAME = r"[A-Z][A-Za-z0-9&'-]*(?:\s+[A-Z][A-Za-z0-9&'-]*){0,2}"
_RELATIONS = (
    (re.compile(rf"(?P<subject>{_NAME})\s+(?P<predicate>(?i:works at|works for))\s+(?P<object>{_NAME})"), "works_at", False),
    (re.compile(rf"(?P<subject>{_NAME})\s+(?P<predicate>(?i:founded))\s+(?P<object>{_NAME})"), "founded", False),
    (re.compile(rf"(?P<subject>{_NAME})\s+(?i:was founded by|is founded by)\s+(?P<object>{_NAME})"), "founded", True),
    (re.compile(rf"(?P<subject>{_NAME})\s+(?P<predicate>(?i:lives in|is based in|based in|is located in|located in))\s+(?P<object>{_NAME})"), "located_in", False),
    (re.compile(rf"(?P<subject>{_NAME})\s+(?P<predicate>(?i:develops|developed|creates|created|builds|built|produces|manufactures))\s+(?P<object>{_NAME})"), "develops", False),
    (re.compile(rf"(?P<subject>{_NAME})\s+(?P<predicate>(?i:was developed by|was created by|was built by))\s+(?P<object>{_NAME})"), "develops", True),
    (re.compile(rf"(?P<subject>{_NAME})\s+(?P<predicate>(?i:partnered with|partners with|collaborated with|collaborates with))\s+(?P<object>{_NAME})"), "partnered_with", False),
    (re.compile(rf"(?P<subject>{_NAME})\s+(?P<predicate>(?i:studies|studied|researches|investigates|focuses on))\s+(?P<object>{_NAME})"), "studies", False),
    (re.compile(rf"(?P<subject>{_NAME})\s+(?P<predicate>(?i:joined|acquired|leads|uses))\s+(?P<object>{_NAME})"), None, False),
)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "the", "to", "was", "were", "what", "when",
    "where", "which", "who", "whom", "why", "with",
}
_ORG_SUFFIXES = {
    "inc", "corp", "corporation", "company", "co", "ltd", "llc", "university", "institute",
    "agency", "bank", "hospital", "innovations", "labs", "laboratory", "research",
}
_LOCATIONS = {"paris", "london", "tokyo", "berlin", "rome", "new york", "san francisco", "california", "france", "germany", "japan"}
_PRONOUNS = {"he", "she", "they", "it", "we", "i", "this", "that", "the", "company"}
_LEADING_NON_ENTITIES = {"a", "an", "and", "but", "for", "in", "it", "the", "this", "what", "when", "where", "which", "who"}
_RELATION_TERMS = {
    "founded": {"found", "founded", "founder", "founded_by"},
    "develops": {"develop", "develops", "developed", "build", "built", "create", "created", "product"},
    "partnered_with": {"partner", "partners", "partnered", "collaborate", "collaborated", "collaboration"},
    "located_in": {"where", "located", "location", "based"},
    "studies": {"study", "studies", "studied", "research", "researches", "investigate", "investigates"},
    "works_at": {"work", "works", "job"},
    "uses": {"use", "uses", "technology"},
}
_GENERIC_QUERY_TERMS = {"product", "hospital", "organization", "person", "research", "collaboration", "topic"}


def _page_for_offset(offset: int, pages: Sequence[Dict[str, Any]]) -> Optional[int]:
    for page in pages:
        if page["start"] <= offset < page["end"]:
            return page["page"]
    return None


def _sentence_rows(text: str, pages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for match in _SENTENCE.finditer(text):
        excerpt = match.group().strip()
        if excerpt:
            rows.append({
                "text": excerpt,
                "start": match.start() + len(match.group()) - len(match.group().lstrip()),
                "end": match.start() + len(match.group().rstrip()),
                "page": _page_for_offset(match.start(), pages),
            })
    return rows


def source_sentence_at(text: str, offset: int) -> Dict[str, Any]:
    """Return the original sentence and offsets containing a source character."""
    for row in _sentence_rows(text, []):
        if row["start"] <= offset < row["end"]:
            return row
    return {"text": "", "start": offset, "end": offset, "page": None}


def _entity_type(name: str, role: Optional[str] = None) -> str:
    normalized = name.lower().strip(" .")
    last = normalized.rsplit(" ", 1)[-1].rstrip(".")
    if normalized in _LOCATIONS or any(normalized.endswith(f" {location}") for location in _LOCATIONS):
        return "LOCATION"
    if role:
        return role
    if last in _ORG_SUFFIXES:
        return "ORGANIZATION"
    return "PERSON_OR_ENTITY"


def _clean_name(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" .,:;")


def _is_entity_name(value: str) -> bool:
    return value.casefold() not in _PRONOUNS | _STOPWORDS and value.split()[0].casefold() not in _LEADING_NON_ENTITIES


def analyze_with_rules(text: str, pages: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Extract conservative, explicitly rule-matched entities and relations from source text."""
    page_ranges = pages or []
    sentences = _sentence_rows(text, page_ranges)
    entities: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
    candidates: List[Dict[str, Any]] = []
    seen_relations = set()

    for sentence in sentences:
        sentence_text = text[sentence["start"]:sentence["end"]]
        for pattern, fixed_label, reverse in _RELATIONS:
            for match in pattern.finditer(sentence_text):
                subject_start = sentence["start"] + match.start("subject")
                subject_end = sentence["start"] + match.end("subject")
                object_start = sentence["start"] + match.start("object")
                object_end = sentence["start"] + match.end("object")
                left = _clean_name(match.group("subject"))
                right = _clean_name(match.group("object"))
                if not _is_entity_name(left) or not _is_entity_name(right):
                    continue
                subject, obj = (right, left) if reverse else (left, right)
                subject_offsets, object_offsets = (
                    ({"start": object_start, "end": object_end}, {"start": subject_start, "end": subject_end})
                    if reverse else
                    ({"start": subject_start, "end": subject_end}, {"start": object_start, "end": object_end})
                )
                label = fixed_label or match.group("predicate").lower().replace(" ", "_")
                subject_role = "PERSON_OR_ENTITY" if reverse and label == "founded" else None
                object_role = (
                    "ORGANIZATION" if label == "works_at"
                    else "LOCATION" if label == "located_in"
                    else "PRODUCT" if label == "develops"
                    else "RESEARCH_TOPIC" if label == "studies"
                    else "TECHNOLOGY" if label == "uses"
                    else None
                )
                subject_type = _entity_type(subject, subject_role)
                object_type = _entity_type(obj, object_role)
                key = (subject.casefold(), label, obj.casefold())
                if key in seen_relations:
                    continue
                seen_relations.add(key)
                entities[(subject_offsets["start"], subject_offsets["end"], subject)] = {
                    "text": subject, "label": subject_type, "start": subject_offsets["start"], "end": subject_offsets["end"],
                }
                entities[(object_offsets["start"], object_offsets["end"], obj)] = {
                    "text": obj, "label": object_type, "start": object_offsets["start"], "end": object_offsets["end"],
                }
                candidates.append({
                    "subject": subject,
                    "object": obj,
                    "label": label,
                    "score": 0.85,
                    "score_type": "explicit_rule_match",
                    "relation_type": "candidate",
                    "subject_label": subject_type,
                    "object_label": object_type,
                    "status": "extracted_candidate",
                    "evidence_snippet": sentence["text"],
                    "source_offsets": {"start": sentence["start"], "end": sentence["end"]},
                    "subject_offsets": subject_offsets,
                    "object_offsets": object_offsets,
                    "page": sentence["page"],
                    "requires_human_review": True,
                    "extraction_method": "lightweight_rule_based",
                })

    role_by_name = {
        entity["text"].casefold(): entity["label"]
        for entity in entities.values()
        if entity["label"] not in {"PERSON_OR_ENTITY", "ENTITY"}
    }
    for match in re.finditer(_NAME, text):
        name = _clean_name(match.group())
        if not _is_entity_name(name):
            continue
        key = (match.start(), match.end(), name)
        entities.setdefault(key, {
            "text": name,
            "label": role_by_name.get(name.casefold(), _entity_type(name)),
            "start": match.start(),
            "end": match.end(),
            "page": _page_for_offset(match.start(), page_ranges),
        })

    for entity in entities.values():
        entity.setdefault("page", _page_for_offset(entity["start"], page_ranges))

    summary = " ".join(row["text"] for row in sentences[:2])
    return {
        "summary": summary,
        "summary_method": "extractive_source_sentences" if summary else None,
        "entities": sorted(entities.values(), key=lambda entity: entity["start"]),
        "candidate_relations": candidates,
        "key_facts": [
            {"text": candidate["evidence_snippet"], "source_offsets": candidate["source_offsets"], "page": candidate["page"],
             "subject": candidate["subject"], "predicate": candidate["label"], "object": candidate["object"],
             "status": "extracted_candidate"}
            for candidate in candidates
        ],
        "extraction_method": "lightweight_rule_based",
    }


def answer_from_document(
    query: str,
    text: str,
    filename: str,
    document_id: str,
    pages: Optional[Sequence[Dict[str, Any]]] = None,
    candidates: Optional[Iterable[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    terms = {
        token for token in re.findall(r"[a-z0-9]+", query.lower())
        if len(token) > 1 and token not in _STOPWORDS
    }
    matches = []
    candidate_list = list(candidates or [])
    relation_intents = {
        label for label, aliases in _RELATION_TERMS.items()
        if any(re.search(rf"\b{re.escape(alias)}\b", query.lower()) for alias in aliases)
    }
    query_entity_terms = terms - _GENERIC_QUERY_TERMS - set().union(*_RELATION_TERMS.values())
    candidate_rows = []
    for candidate in candidate_list:
        label = candidate.get("label", candidate.get("predicate", ""))
        if label not in relation_intents:
            continue
        entity_terms = {
            token for token in re.findall(
                r"[a-z0-9]+", f"{candidate.get('subject', '')} {candidate.get('object', '')}".lower()
            )
            if len(token) > 1
        }
        if not query_entity_terms or entity_terms.intersection(query_entity_terms):
            candidate_rows.append(candidate)

    for row in _sentence_rows(text, pages or []):
        row_terms = set(re.findall(r"[a-z0-9]+", row["text"].lower()))
        overlap = terms & row_terms
        coverage = len(overlap) / max(1, len(terms))
        if (overlap and coverage >= 0.5) or any(
            candidate.get("evidence_snippet") == row["text"] for candidate in candidate_rows
        ):
            matches.append((coverage, row))
    matches.sort(key=lambda item: (-item[0], item[1]["start"]))
    evidence = [
        {
            "excerpt": row["text"],
            "source": filename or document_id,
            "document_id": document_id,
            "page": row["page"],
            "source_offsets": {"start": row["start"], "end": row["end"]},
        }
        for _, row in matches[:3]
    ]
    if not evidence:
        answer = "I couldn't find supporting evidence for that question in this document."
    else:
        answer = "The document says: " + " ".join(item["excerpt"] for item in evidence)

    relevant_terms = terms
    relationships = [
        {
            "subject": candidate["subject"],
            "predicate": candidate["label"],
            "object": candidate["object"],
            "status": candidate.get("status", "extracted_candidate"),
            "evidence": candidate["evidence_snippet"],
            "page": candidate.get("page"),
            "document_id": document_id,
        }
        for candidate in candidate_list
        if candidate in candidate_rows or relevant_terms.intersection(
            set(re.findall(r"[a-z0-9]+", f"{candidate['subject']} {candidate['label']} {candidate['object']}".lower()))
        )
    ]
    return {
        "answer": answer,
        "uncertain": not bool(evidence),
        "evidence": evidence,
        "entities": sorted({entity for relation in relationships for entity in (relation["subject"], relation["object"])}),
        "relationships": relationships,
    }
