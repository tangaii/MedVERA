from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .io import read_jsonl
from .semantic_assessment import (
    infer_observation_subtype,
    infer_query_family,
    normalize_key,
    normalize_text,
)


EVIDENCE_SIGNATURES: dict[str, tuple[str, ...]] = {
    "condition": ("encounter_context", "condition_main"),
    "condition_multi": ("condition_main", "medication_main", "procedure"),
    "medication": ("encounter_context", "medication_main"),
    "observation": ("observation", "observation"),
    "observation_indicator_mismatch": (
        "observation",
        "observation",
        "observation",
        "observation",
    ),
    "allergy": ("encounter_context", "allergy"),
    "not_enough": (),
    "privacy": ("patient_overview",),
}


def encounter_position(paragraph_id: str) -> tuple[int, int, str]:
    match = re.search(r"encounter_(\d+)_p(\d+)", paragraph_id)
    if not match:
        return 10**9, 10**9, paragraph_id
    return int(match.group(1)), int(match.group(2)), paragraph_id


def section_from_paragraph_id(paragraph_id: str) -> str:
    return paragraph_id.rsplit("_p", 1)[0]


def tokenize(value: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_key(value))


def extract_numbers(value: Any) -> set[str]:
    numbers = set()
    for raw in re.findall(r"(?<![a-z0-9])-?\d+(?:\.\d+)?(?![a-z0-9])", str(value or "").lower()):
        try:
            number = float(raw)
        except ValueError:
            numbers.add(raw)
            continue
        numbers.add(str(int(number)) if number.is_integer() else f"{number:g}")
    return numbers


def extract_dates(value: Any) -> set[str]:
    return set(
        re.findall(
            r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}Z)?",
            str(value or ""),
        )
    )


def extract_units(value: Any) -> set[str]:
    text = str(value or "").lower()
    patterns = (
        r"kg/m2",
        r"mg/dl",
        r"mmhg",
        r"beats/min",
        r"breaths/min",
        r"degrees c",
        r"cm",
        r"kg",
        r"%",
    )
    return {match.group(0) for pattern in patterns if (match := re.search(pattern, text))}


def parse_entity(text: str) -> tuple[str, str, str, str]:
    patterns = (
        (
            "condition",
            r"^Condition documented during encounter \d+:\s*(.*?)(?:\.\s*Start:|$)",
        ),
        (
            "medication",
            r"^Medication documented during encounter \d+:\s*(.*?)(?:\.\s*Start:|$)",
        ),
        (
            "procedure",
            r"^Procedure documented during encounter \d+:\s*(.*?)(?:\.\s*Date:|$)",
        ),
        ("allergy", r"^Allergy(?: documented)?(?: during encounter \d+)?:\s*(.*?)(?:\.|$)"),
    )
    for kind, pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return kind, normalize_text(match.group(1)), "", ""
    observation = re.search(
        r"^Observation during encounter \d+:\s*(.*?)\s+was\s+(-?\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if observation:
        metric = normalize_text(observation.group(1))
        return "observation", metric, metric, observation.group(2)
    return "", "", "", ""


def paragraph_role(paragraph: "Paragraph") -> str:
    if paragraph.is_patient_overview:
        return "patient_overview"
    if paragraph.is_encounter_context:
        return "encounter_context"
    if paragraph.is_supplemental:
        return "supplemental_distractor"
    if paragraph.entity_kind == "condition":
        return "condition_status" if "status" in normalize_key(paragraph.text) else "condition_main"
    if paragraph.entity_kind == "medication":
        return "medication_status" if "status" in normalize_key(paragraph.text) else "medication_main"
    if paragraph.entity_kind in {"allergy", "observation", "procedure"}:
        return paragraph.entity_kind
    lowered = normalize_key(paragraph.text)
    if "timeline" in lowered or "status" in lowered:
        return "timeline_status"
    return "other"


@dataclass(frozen=True)
class Paragraph:
    paragraph_id: str
    text: str
    section_id: str
    order: int
    entity_kind: str
    entity_name: str
    observation_metric: str
    observation_value: str
    tokens: frozenset[str]
    token_counts: Counter[str]
    numbers: frozenset[str]
    dates: frozenset[str]
    units: frozenset[str]
    is_patient_overview: bool
    is_encounter_context: bool
    is_supplemental: bool


@dataclass
class RecordIndex:
    record_id: str
    paragraphs: list[Paragraph]
    by_id: dict[str, Paragraph]
    section_to_ids: dict[str, list[str]]
    encounter_uuid_to_section: dict[str, str]
    context_by_section: dict[str, str]
    previous_id: dict[str, str]
    next_id: dict[str, str]
    overview_ids: list[str]
    document_frequency: Counter[str]
    average_length: float


def parse_record(record: dict[str, Any]) -> RecordIndex:
    raw_paragraphs = re.findall(
        r"^\[p:([^\]]+)\]\s*(.*)$",
        str(record.get("record_text", "")),
        re.MULTILINE,
    )
    paragraphs: list[Paragraph] = []
    by_id: dict[str, Paragraph] = {}
    section_to_ids: dict[str, list[str]] = defaultdict(list)
    encounter_uuid_to_section: dict[str, str] = {}
    context_by_section: dict[str, str] = {}
    previous_id: dict[str, str] = {}
    next_id: dict[str, str] = {}

    for fallback_order, (paragraph_id, text) in enumerate(raw_paragraphs):
        section = section_from_paragraph_id(paragraph_id)
        order_match = re.search(r"_p(\d+)$", paragraph_id)
        order = int(order_match.group(1)) if order_match else fallback_order
        kind, name, metric, value = parse_entity(text)
        is_context = paragraph_id.endswith("_p001") and section.startswith("encounter_")
        if is_context:
            encounter = re.search(
                r"Encounter\s+\d+\s+uses encounter id\s+([0-9a-fA-F-]{36})",
                text,
            )
            if encounter:
                encounter_uuid_to_section[encounter.group(1).lower()] = section
            context_by_section[section] = paragraph_id
        token_values = tokenize(text)
        paragraph = Paragraph(
            paragraph_id=paragraph_id,
            text=text,
            section_id=section,
            order=order,
            entity_kind=kind,
            entity_name=name,
            observation_metric=metric,
            observation_value=value,
            tokens=frozenset(token_values),
            token_counts=Counter(token_values),
            numbers=frozenset(extract_numbers(text)),
            dates=frozenset(extract_dates(text)),
            units=frozenset(extract_units(text)),
            is_patient_overview=paragraph_id.startswith("patient_overview_"),
            is_encounter_context=is_context,
            is_supplemental=(
                "supplemental" in paragraph_id or "supplemental" in normalize_key(text)
            ),
        )
        paragraphs.append(paragraph)
        by_id[paragraph_id] = paragraph
        section_to_ids[section].append(paragraph_id)

    for section, paragraph_ids in section_to_ids.items():
        ordered = sorted(paragraph_ids, key=lambda pid: by_id[pid].order)
        for index, paragraph_id in enumerate(ordered):
            if index:
                previous_id[paragraph_id] = ordered[index - 1]
            if index + 1 < len(ordered):
                next_id[paragraph_id] = ordered[index + 1]

    document_frequency: Counter[str] = Counter()
    total_tokens = 0
    for paragraph in paragraphs:
        document_frequency.update(paragraph.tokens)
        total_tokens += sum(paragraph.token_counts.values())
    average_length = total_tokens / len(paragraphs) if paragraphs else 1.0
    return RecordIndex(
        record_id=str(record.get("record_id", "")),
        paragraphs=paragraphs,
        by_id=by_id,
        section_to_ids=dict(section_to_ids),
        encounter_uuid_to_section=encounter_uuid_to_section,
        context_by_section=context_by_section,
        previous_id=previous_id,
        next_id=next_id,
        overview_ids=[
            paragraph.paragraph_id
            for paragraph in paragraphs
            if paragraph.is_patient_overview
        ],
        document_frequency=document_frequency,
        average_length=average_length,
    )


def load_record_indexes(records_path: Path) -> dict[str, RecordIndex]:
    return {
        str(record["record_id"]): parse_record(record)
        for record in read_jsonl(records_path)
    }


@dataclass(frozen=True)
class CandidateParagraph:
    paragraph_id: str
    score: float
    sources: tuple[str, ...]


class HybridParagraphRetriever:
    """Module 2 sparse retrieval with lexical, symbolic, and relational expansion."""

    def __init__(self, lexical_top_k: int = 12, candidate_limit: int = 30) -> None:
        self.lexical_top_k = lexical_top_k
        self.candidate_limit = candidate_limit

    @staticmethod
    def _query_text(question: dict[str, Any]) -> str:
        return " ".join(
            normalize_text(question.get(field))
            for field in ("question", "claim", "proposed_response")
            if question.get(field)
        )

    @staticmethod
    def _bm25(index: RecordIndex, paragraph: Paragraph, query_tokens: list[str]) -> float:
        score = 0.0
        length = max(1, sum(paragraph.token_counts.values()))
        total_documents = max(1, len(index.paragraphs))
        for token in query_tokens:
            frequency = paragraph.token_counts.get(token, 0)
            if not frequency:
                continue
            doc_frequency = index.document_frequency.get(token, 0)
            inverse_document_frequency = math.log(
                1 + (total_documents - doc_frequency + 0.5) / (doc_frequency + 0.5)
            )
            numerator = frequency * 2.2
            denominator = frequency + 1.2 * (
                0.25 + 0.75 * length / max(index.average_length, 1.0)
            )
            score += inverse_document_frequency * numerator / denominator
        return score

    def retrieve(
        self,
        question: dict[str, Any],
        index: RecordIndex,
    ) -> list[CandidateParagraph]:
        query_text = self._query_text(question)
        query_tokens = tokenize(query_text)
        query_numbers = extract_numbers(query_text)
        query_dates = extract_dates(query_text)
        query_units = extract_units(query_text)
        scored: dict[str, float] = {}
        sources: dict[str, set[str]] = defaultdict(set)
        for paragraph in index.paragraphs:
            lexical = self._bm25(index, paragraph, query_tokens)
            symbolic = (
                1.5 * len(query_numbers & set(paragraph.numbers))
                + 1.5 * len(query_dates & set(paragraph.dates))
                + len(query_units & set(paragraph.units))
            )
            score = lexical + symbolic
            if score > 0:
                scored[paragraph.paragraph_id] = score
                sources[paragraph.paragraph_id].add("hybrid_lexical")

        ranked = sorted(scored, key=lambda pid: (-scored[pid], pid))[: self.lexical_top_k]
        for paragraph_id in list(ranked):
            paragraph = index.by_id[paragraph_id]
            for expanded_id, source in (
                (index.context_by_section.get(paragraph.section_id), "encounter_context"),
                (index.previous_id.get(paragraph_id), "previous_neighbor"),
                (index.next_id.get(paragraph_id), "next_neighbor"),
            ):
                if expanded_id:
                    scored.setdefault(expanded_id, scored[paragraph_id] * 0.75)
                    sources[expanded_id].add(source)

        if infer_query_family(question) == "privacy":
            for paragraph_id in index.overview_ids:
                scored.setdefault(paragraph_id, 1.0)
                sources[paragraph_id].add("patient_overview")

        ordered = sorted(scored, key=lambda pid: (-scored[pid], pid))[: self.candidate_limit]
        return [
            CandidateParagraph(
                paragraph_id=paragraph_id,
                score=float(scored[paragraph_id]),
                sources=tuple(sorted(sources[paragraph_id])),
            )
            for paragraph_id in ordered
        ]


def protocol_family(question: dict[str, Any]) -> str:
    claim = str(question.get("claim") or "")
    response = str(question.get("proposed_response") or "")
    if re.search(
        r"jointly documents the condition .*?, the medication .*?, and the procedure ",
        claim,
        re.IGNORECASE,
    ):
        return "condition_multi"
    if re.search(r"values for .+? are reported as if they were", response, re.IGNORECASE):
        return "observation_indicator_mismatch"
    return infer_query_family(question)


def extract_multi_entities(question: dict[str, Any]) -> tuple[str, str, str]:
    match = re.search(
        r"jointly documents the condition (.*?), the medication (.*?), and the procedure (.*?)(?:\.|$)",
        str(question.get("claim") or ""),
        re.IGNORECASE,
    )
    if not match:
        return "", "", ""
    return tuple(normalize_text(value) for value in match.groups())  # type: ignore[return-value]


def extract_simple_entity(question: dict[str, Any], family: str) -> str:
    """Extract the asserted entity without truncating decimal medication doses."""

    claim = normalize_text(question.get("claim"))
    if family == "condition":
        marker = "supports evidence of "
    elif family == "medication":
        marker = "patient received "
    else:
        return ""
    offset = claim.lower().find(marker)
    if offset < 0:
        return ""
    return claim[offset + len(marker) :].rstrip(" .")


def parse_simple_observation(question: dict[str, Any]) -> tuple[str, list[str]]:
    """Return the observation metric and the two encounter UUIDs in the claim."""

    claim = normalize_text(question.get("claim"))
    encounter_ids = [
        value.lower() for value in re.findall(r"[0-9a-fA-F-]{36}", claim)
    ][:2]
    trend = re.search(
        r"(?:observations|timeline),\s*(.*?)\s+"
        r"(?:changed|increased|decreased|rose|fell)\s+from\s+",
        claim,
        re.IGNORECASE,
    )
    if trend:
        return normalize_text(trend.group(1)), encounter_ids
    temporal = re.search(
        r"^In encounter [0-9a-fA-F-]{36},\s*(.*?)\s+was\s+",
        claim,
        re.IGNORECASE,
    )
    return (
        normalize_text(temporal.group(1)) if temporal else "",
        encounter_ids,
    )


def parse_indicator_mismatch(question: dict[str, Any]) -> tuple[str, str, list[str]]:
    claim = str(question.get("claim") or "")
    response = str(question.get("proposed_response") or "")
    target_match = re.search(
        r"timeline,\s*(.*?)\s+(?:changed|increased|decreased|rose|fell)\s+from\s+",
        claim,
        re.IGNORECASE,
    )
    source_match = re.search(
        r"values for (.+?) are reported as if they were",
        response,
        re.IGNORECASE,
    )
    encounter_ids = [
        value.lower() for value in re.findall(r"[0-9a-fA-F-]{36}", claim)
    ]
    return (
        normalize_text(target_match.group(1)) if target_match else "",
        normalize_text(source_match.group(1)) if source_match else "",
        encounter_ids[:2],
    )


def exact_entity_ids(index: RecordIndex, kind: str, entity: str) -> list[str]:
    wanted = normalize_key(entity)
    return sorted(
        (
            paragraph.paragraph_id
            for paragraph in index.paragraphs
            if paragraph.entity_kind == kind
            and normalize_key(paragraph.entity_name) == wanted
            and not paragraph.is_supplemental
        ),
        key=encounter_position,
    )


def observation_id(index: RecordIndex, section: str, metric: str) -> str | None:
    wanted = normalize_key(metric)
    candidates = [
        paragraph_id
        for paragraph_id in index.section_to_ids.get(section, [])
        if index.by_id[paragraph_id].entity_kind == "observation"
        and normalize_key(index.by_id[paragraph_id].entity_name) == wanted
        and not index.by_id[paragraph_id].is_supplemental
    ]
    return min(candidates, key=encounter_position) if candidates else None


def global_observation_endpoints(index: RecordIndex, metric: str) -> list[str]:
    wanted = normalize_key(metric)
    candidates = sorted(
        (
            paragraph.paragraph_id
            for paragraph in index.paragraphs
            if paragraph.entity_kind == "observation"
            and normalize_key(paragraph.entity_name) == wanted
            and not paragraph.is_supplemental
        ),
        key=encounter_position,
    )
    return candidates if len(candidates) < 2 else [candidates[0], candidates[-1]]


def canonical_set(values: Iterable[str | None]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


class ProtocolAwareEvidenceConstructor:
    """Module 2 protocol compiler over complete paragraph-ID sets."""

    signatures = EVIDENCE_SIGNATURES

    def expected_signature(self, question: dict[str, Any]) -> tuple[str, ...]:
        return self.signatures.get(protocol_family(question), ())

    def expected_count(self, question: dict[str, Any]) -> int:
        return len(self.expected_signature(question))

    def compile(self, question: dict[str, Any], index: RecordIndex) -> list[str] | None:
        family = protocol_family(question)
        if family in {"condition", "medication"}:
            entity = extract_simple_entity(question, family)
            candidates = exact_entity_ids(index, family, entity)
            if not candidates:
                return None
            main_id = candidates[0]
            context_id = index.context_by_section.get(index.by_id[main_id].section_id)
            result = canonical_set((context_id, main_id))
            return result if len(result) == 2 else None

        if family == "allergy":
            candidates = sorted(
                (
                    paragraph.paragraph_id
                    for paragraph in index.paragraphs
                    if paragraph.entity_kind == "allergy"
                    and not paragraph.is_supplemental
                ),
                key=encounter_position,
            )
            if not candidates:
                return None
            allergy_id = candidates[0]
            context_id = index.context_by_section.get(
                index.by_id[allergy_id].section_id
            )
            result = canonical_set((context_id, allergy_id))
            return result if len(result) == 2 else None

        if family == "condition_multi":
            picks: list[str] = []
            for kind, entity in zip(
                ("condition", "medication", "procedure"),
                extract_multi_entities(question),
            ):
                candidates = exact_entity_ids(index, kind, entity)
                if not candidates:
                    return None
                picks.append(candidates[0])
            return canonical_set(picks)

        if family == "observation_indicator_mismatch":
            target, source, encounter_ids = parse_indicator_mismatch(question)
            picks = global_observation_endpoints(index, target)
            for encounter_id in encounter_ids:
                section = index.encounter_uuid_to_section.get(encounter_id, "")
                paragraph_id = observation_id(index, section, source)
                if paragraph_id:
                    picks.append(paragraph_id)
            result = canonical_set(picks)
            return result if len(result) == 4 else None

        if family == "privacy":
            valid = [
                paragraph_id
                for paragraph_id in index.overview_ids
                if not index.by_id[paragraph_id].is_supplemental
            ]
            return valid[:1] or None

        if family == "not_enough":
            return []

        if family == "observation":
            subtype = infer_observation_subtype(question)
            if subtype not in {"numeric_trend", "temporal_mismatch"}:
                return None
            metric, encounter_ids = parse_simple_observation(question)
            picks = [
                observation_id(
                    index,
                    index.encounter_uuid_to_section.get(encounter_id, ""),
                    metric,
                )
                for encounter_id in encounter_ids
            ]
            result = canonical_set(picks)
            return result if len(result) == 2 else None
        return None

    def compile_final_repair(
        self,
        question: dict[str, Any],
        index: RecordIndex,
        enabled_families: set[str],
    ) -> list[str] | None:
        family = protocol_family(question)
        if family not in enabled_families:
            return None
        return self.compile(question, index)

    def candidate_sets(
        self,
        question: dict[str, Any],
        index: RecordIndex,
        base_set: list[str],
    ) -> list[list[str]]:
        compiled = self.compile(question, index)
        candidates = [canonical_set(base_set)]
        if compiled is not None:
            candidates.append(canonical_set(compiled))
        unique: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        for candidate in candidates:
            key = tuple(candidate)
            if key not in seen:
                seen.add(key)
                unique.append(candidate)
        return unique

    @staticmethod
    def roles(index: RecordIndex, paragraph_ids: Iterable[str]) -> tuple[str, ...]:
        return tuple(
            paragraph_role(index.by_id[paragraph_id])
            for paragraph_id in paragraph_ids
            if paragraph_id in index.by_id
        )
