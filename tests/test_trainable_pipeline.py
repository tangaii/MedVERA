from __future__ import annotations

import sys
import tempfile
import unittest
import warnings
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medvera.evidence_construction import parse_record
from medvera.io import public_view
from medvera.training import TrainableMedVERA, TrainingConfig


def _record(record_id: str, encounter_id: str) -> dict[str, str]:
    return {
        "record_id": record_id,
        "record_text": "\n".join(
            (
                "[p:patient_overview_p001] Synthetic patient overview.",
                (
                    "[p:encounter_001_p001] Encounter 1 uses encounter id "
                    f"{encounter_id}. Date: 2025-01-01."
                ),
                (
                    "[p:encounter_001_p002] Condition documented during encounter 1: "
                    "Asthma (disorder). Start: 2025-01-01."
                ),
                (
                    "[p:encounter_002_p001] Encounter 2 uses encounter id "
                    "22222222-2222-4222-8222-222222222222. Date: 2025-02-01."
                ),
                (
                    "[p:encounter_002_p002] Condition documented during encounter 2: "
                    "Diabetes mellitus (disorder). Start: 2025-02-01."
                ),
            )
        ),
    }


def _question(
    qid: str,
    record_id: str,
    trusted: bool,
) -> dict[str, object]:
    entity = "Asthma (disorder)" if trusted else "Diabetes mellitus (disorder)"
    return {
        "qid": qid,
        "record_id": record_id,
        "question": "Is the response supported by the longitudinal record?",
        "claim": f"The longitudinal patient record supports evidence of {entity}.",
        "proposed_response": None if trusted else f"The record supports {entity}.",
        "options": {"A": "trusted", "B": "untrusted"},
        "answer_format": "json",
        "target": {
            "trust_label": "trusted" if trusted else "untrusted",
            "risk_types": [] if trusted else ["hallucination.fact_mismatch"],
            "evidence": [
                {"paragraph_id": "encounter_001_p001"},
                {"paragraph_id": "encounter_001_p002"},
            ],
        },
    }


class TrainablePipelineTest(unittest.TestCase):
    def test_fit_save_load_and_predict_from_raw_annotations(self) -> None:
        records = [
            _record("record_a", "11111111-1111-4111-8111-111111111111"),
            _record("record_b", "33333333-3333-4333-8333-333333333333"),
        ]
        indexes = {row["record_id"]: parse_record(row) for row in records}
        questions = [
            _question("a_trusted", "record_a", True),
            _question("a_untrusted", "record_a", False),
            _question("b_trusted", "record_b", True),
            _question("b_untrusted", "record_b", False),
        ]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Label not .*")
            model = TrainableMedVERA(
                TrainingConfig(
                    seeds=(7,),
                    folds=2,
                    consensus_min_support=1,
                    paragraph_candidate_limit=16,
                )
            ).fit(questions, indexes)
        prediction = model.predict(questions, indexes)
        for question in questions:
            row = prediction[str(question["qid"])]
            self.assertEqual(row["trust_label"], question["target"]["trust_label"])
            self.assertEqual(row["risk_types"], question["target"]["risk_types"])
            self.assertEqual(row["evidence"], question["target"]["evidence"])

        with tempfile.TemporaryDirectory(prefix="medvera_trainable_") as temporary:
            model_dir = Path(temporary)
            model.save(model_dir)
            restored = TrainableMedVERA.load(model_dir)
            self.assertEqual(
                restored.predict(
                    [public_view(question) for question in questions],
                    indexes,
                ),
                prediction,
            )


if __name__ == "__main__":
    unittest.main()
