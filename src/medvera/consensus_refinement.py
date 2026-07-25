from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from .io import ZERO_TOKEN_SUMMARY, evidence_ids


@dataclass(frozen=True)
class ConsensusDecision:
    accepted: bool
    support_count: int
    required_support: int
    selected_set: list[str]
    reason: str


class ConsensusEvidenceRefiner:
    """Module 4 conservative 17-of-18 evidence-set acceptance gate."""

    def __init__(self, unit_count: int = 18, required_support: int = 17) -> None:
        if not 0 < required_support <= unit_count:
            raise ValueError("required_support must be between 1 and unit_count")
        self.unit_count = unit_count
        self.required_support = required_support

    def decide(
        self,
        base_set: list[str],
        candidate_set: list[str],
        unit_votes: list[bool],
        protocol_complete: bool,
    ) -> ConsensusDecision:
        if len(unit_votes) != self.unit_count:
            raise ValueError(
                f"expected {self.unit_count} fold-seed votes, got {len(unit_votes)}"
            )
        support_count = sum(unit_votes)
        accepted = protocol_complete and support_count >= self.required_support
        return ConsensusDecision(
            accepted=accepted,
            support_count=support_count,
            required_support=self.required_support,
            selected_set=list(candidate_set if accepted else base_set),
            reason=(
                "consensus_candidate"
                if accepted
                else "incomplete_protocol"
                if not protocol_complete
                else "insufficient_consensus"
            ),
        )

    @staticmethod
    def apply_evidence(
        base_prediction: dict[str, Any],
        evidence_by_qid: dict[str, list[str]],
    ) -> dict[str, Any]:
        result = copy.deepcopy(base_prediction)
        for qid, paragraph_ids in evidence_by_qid.items():
            if qid not in result:
                raise KeyError(f"unknown qid in evidence refinement: {qid}")
            result[qid]["evidence"] = [
                {"paragraph_id": paragraph_id} for paragraph_id in paragraph_ids
            ]
        result["summary"] = copy.deepcopy(ZERO_TOKEN_SUMMARY)
        return result

    @staticmethod
    def evidence_cardinality(prediction: dict[str, Any]) -> int:
        return sum(
            len(evidence_ids(row))
            for qid, row in prediction.items()
            if qid != "summary" and isinstance(row, dict)
        )

    def batch_accept(
        self,
        previous_prediction: dict[str, Any],
        candidate_prediction: dict[str, Any],
        expected_cardinality: int | None,
    ) -> tuple[dict[str, Any], bool]:
        if expected_cardinality is None:
            return candidate_prediction, True
        if self.evidence_cardinality(candidate_prediction) == expected_cardinality:
            return candidate_prediction, True
        return previous_prediction, False
