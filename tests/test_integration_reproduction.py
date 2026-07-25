from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medvera.evidence_construction import load_record_indexes
from medvera.pipeline import MedVERAPipeline
from medvera.io import read_jsonl, sha256_file, write_json
from medvera.training import TrainableMedVERA


class IntegrationReproductionTest(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("MEDVERA_DATA_DIR"),
        "set MEDVERA_DATA_DIR to run end-to-end reproduction",
    )
    def test_winning_submission_hash(self) -> None:
        data_dir = Path(os.environ["MEDVERA_DATA_DIR"])
        pipeline = MedVERAPipeline(ROOT, data_dir)
        result = pipeline.reproduce("test")
        with tempfile.TemporaryDirectory(prefix="medvera_test_") as temporary:
            output = Path(temporary) / "submission.json"
            write_json(output, result.prediction)
            self.assertEqual(
                sha256_file(output),
                "7ed85f5c47aad1b3e9f28abf89a044353b386cfc873d1ac4b8faf7d49f5e6daa",
            )

    @unittest.skipUnless(
        os.environ.get("MEDVERA_DATA_DIR")
        and os.environ.get("MEDVERA_RUN_FULL_TRAINING") == "1",
        "set MEDVERA_DATA_DIR and MEDVERA_RUN_FULL_TRAINING=1",
    )
    def test_raw_training_matches_dev_and_winning_test_fields(self) -> None:
        data_dir = Path(os.environ["MEDVERA_DATA_DIR"])
        train_questions = read_jsonl(data_dir / "train" / "questions.jsonl")
        train_indexes = load_record_indexes(data_dir / "train" / "records.jsonl")
        model = TrainableMedVERA().fit(train_questions, train_indexes)

        dev_questions = read_jsonl(data_dir / "dev" / "questions.jsonl")
        dev_indexes = load_record_indexes(data_dir / "dev" / "records.jsonl")
        dev_prediction = model.predict(dev_questions, dev_indexes)
        for question in dev_questions:
            predicted = dev_prediction[str(question["qid"])]
            target = question["target"]
            self.assertEqual(predicted["trust_label"], target["trust_label"])
            self.assertEqual(set(predicted["risk_types"]), set(target["risk_types"]))
            self.assertEqual(
                {item["paragraph_id"] for item in predicted["evidence"]},
                {item["paragraph_id"] for item in target["evidence"]},
            )

        test_questions = read_jsonl(data_dir / "test" / "questions.jsonl")
        test_indexes = load_record_indexes(data_dir / "test" / "records.jsonl")
        trained = model.predict(test_questions, test_indexes)
        frozen = MedVERAPipeline(ROOT, data_dir).reproduce("test").prediction
        for question in test_questions:
            qid = str(question["qid"])
            for field in ("trust_label", "risk_types", "evidence"):
                self.assertEqual(trained[qid][field], frozen[qid][field])


if __name__ == "__main__":
    unittest.main()
