from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medvera.evidence_construction import (
    ProtocolAwareEvidenceConstructor,
    parse_record,
)


class EvidenceConstructionTest(unittest.TestCase):
    def test_multi_fact_condition_protocol(self) -> None:
        record = {
            "record_id": "record-1",
            "record_text": "\n".join(
                (
                    "[p:encounter_002_p001] Encounter 2 uses encounter id 00000000-0000-0000-0000-000000000002.",
                    "[p:encounter_002_p002] Condition documented during encounter 2: Childhood asthma. Start: 2020-01-01.",
                    "[p:encounter_002_p003] Medication documented during encounter 2: Albuterol. Start: 2020-01-01.",
                    "[p:encounter_002_p004] Procedure documented during encounter 2: Spirometry. Date: 2020-01-01.",
                )
            ),
        }
        question = {
            "claim": (
                "The record jointly documents the condition Childhood asthma, "
                "the medication Albuterol, and the procedure Spirometry."
            ),
            "proposed_response": "The three facts are jointly documented.",
        }
        constructor = ProtocolAwareEvidenceConstructor()
        self.assertEqual(
            constructor.compile(question, parse_record(record)),
            [
                "encounter_002_p002",
                "encounter_002_p003",
                "encounter_002_p004",
            ],
        )

    def test_indicator_mismatch_protocol_preserves_role_order(self) -> None:
        first = "00000000-0000-0000-0000-000000000001"
        second = "00000000-0000-0000-0000-000000000004"
        record = {
            "record_id": "record-2",
            "record_text": "\n".join(
                (
                    f"[p:encounter_001_p001] Encounter 1 uses encounter id {first}.",
                    "[p:encounter_001_p002] Observation during encounter 1: Glucose was 80 mg/dL.",
                    "[p:encounter_001_p003] Observation during encounter 1: Body Weight was 70 kg.",
                    f"[p:encounter_004_p001] Encounter 4 uses encounter id {second}.",
                    "[p:encounter_004_p002] Observation during encounter 4: Glucose was 95 mg/dL.",
                    "[p:encounter_004_p003] Observation during encounter 4: Body Weight was 72 kg.",
                )
            ),
        }
        question = {
            "claim": (
                f"Across the rendered timeline, Glucose changed from 80 to 95 "
                f"in encounters {first} and {second}."
            ),
            "proposed_response": "The values for Body Weight are reported as if they were Glucose.",
        }
        constructor = ProtocolAwareEvidenceConstructor()
        self.assertEqual(
            constructor.compile(question, parse_record(record)),
            [
                "encounter_001_p002",
                "encounter_004_p002",
                "encounter_001_p003",
                "encounter_004_p003",
            ],
        )


if __name__ == "__main__":
    unittest.main()
