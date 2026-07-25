from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medvera.semantic_assessment import (
    GroundedResidualVerifier,
    infer_observation_subtype,
    infer_query_family,
)


class SemanticAssessmentTest(unittest.TestCase):
    def test_public_family_routing(self) -> None:
        self.assertEqual(
            infer_query_family(
                {
                    "question": "Does this satisfy privacy requirements?",
                    "claim": "The response exposes a synthetic patient identifier.",
                }
            ),
            "privacy",
        )
        self.assertEqual(
            infer_query_family(
                {
                    "claim": "The medication timeline supports that the patient received Albuterol."
                }
            ),
            "medication",
        )

    def test_observation_subtype(self) -> None:
        question = {
            "claim": "Glucose increased from 120 to 90.",
            "proposed_response": "The value increased.",
        }
        self.assertEqual(infer_observation_subtype(question), "numeric_trend")
        self.assertTrue(GroundedResidualVerifier.numeric_direction_conflict(question))


if __name__ == "__main__":
    unittest.main()
