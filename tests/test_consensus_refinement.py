from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medvera.consensus_refinement import ConsensusEvidenceRefiner
from medvera.exact_set_scoring import ensemble_margin_support


class ConsensusRefinementTest(unittest.TestCase):
    def test_fold_seed_vote_requires_positive_sum_and_two_nonnegative_margins(self) -> None:
        self.assertTrue(ensemble_margin_support([0.2, 0.1, -0.05]))
        self.assertFalse(ensemble_margin_support([0.2, -0.1, -0.05]))
        self.assertFalse(ensemble_margin_support([0.0, 0.0, 0.0]))

    def test_consensus_accepts_seventeen_of_eighteen_votes(self) -> None:
        refiner = ConsensusEvidenceRefiner(unit_count=18, required_support=17)
        decision = refiner.decide(
            base_set=["encounter_001_p001"],
            candidate_set=["encounter_001_p001", "encounter_001_p002"],
            unit_votes=[True] * 17 + [False],
            protocol_complete=True,
        )
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.support_count, 17)
        self.assertEqual(
            decision.selected_set,
            ["encounter_001_p001", "encounter_001_p002"],
        )

    def test_consensus_falls_back_when_protocol_is_incomplete(self) -> None:
        refiner = ConsensusEvidenceRefiner(unit_count=18, required_support=17)
        decision = refiner.decide(
            base_set=["base"],
            candidate_set=["candidate"],
            unit_votes=[True] * 18,
            protocol_complete=False,
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.selected_set, ["base"])


if __name__ == "__main__":
    unittest.main()
