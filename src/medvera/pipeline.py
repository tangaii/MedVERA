from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .consensus_refinement import ConsensusEvidenceRefiner
from .evidence_construction import (
    ProtocolAwareEvidenceConstructor,
    load_record_indexes,
)
from .io import ZERO_TOKEN_SUMMARY, evidence_ids, read_json, read_jsonl, write_json


@dataclass(frozen=True)
class ReproductionResult:
    split: str
    prediction: dict[str, Any]
    repaired_qids: tuple[str, ...]
    evidence_count: int
    batch_accepted: bool


class MedVERAPipeline:
    """Stable inference facade matching the four modules described in the paper.

    Exact competition reproduction starts from the frozen outputs of Modules
    1--3 and reruns the deterministic protocol compilation and Module 4 batch
    acceptance. The repository documents this distinction explicitly.
    """

    def __init__(
        self,
        repository_root: Path,
        data_dir: Path,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.data_dir = data_dir.resolve()
        self.artifacts_dir = self.repository_root / "artifacts"
        self.protocol = read_json(self.artifacts_dir / "protocol.json")
        self.constructor = ProtocolAwareEvidenceConstructor()
        validation = self.protocol["validation"]
        self.refiner = ConsensusEvidenceRefiner(
            unit_count=int(validation["fold_seed_units"]),
            required_support=int(validation["required_support"]),
        )

    def _validate_split_files(self, split: str) -> tuple[Path, Path]:
        if split not in {"dev", "test"}:
            raise ValueError("split must be 'dev' or 'test'")
        questions = self.data_dir / split / "questions.jsonl"
        records = self.data_dir / split / "records.jsonl"
        missing = [path for path in (questions, records) if not path.is_file()]
        if missing:
            rendered = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(
                f"official data files are missing: {rendered}. "
                "See the Data section of README.md for the expected layout."
            )
        return questions, records

    def reproduce(self, split: str) -> ReproductionResult:
        questions_path, records_path = self._validate_split_files(split)
        questions = read_jsonl(questions_path)
        indexes = load_record_indexes(records_path)
        base_path = self.artifacts_dir / f"base_{split}.json"
        prediction = copy.deepcopy(read_json(base_path))
        enabled_families = set(self.protocol["repair_families"])
        repaired_qids: list[str] = []

        for question in questions:
            qid = str(question["qid"])
            record_id = str(question["record_id"])
            if qid not in prediction:
                raise KeyError(f"base prediction is missing qid {qid}")
            candidate = self.constructor.compile_final_repair(
                question,
                indexes[record_id],
                enabled_families,
            )
            if candidate is None or candidate == evidence_ids(prediction[qid]):
                continue
            prediction[qid]["evidence"] = [
                {"paragraph_id": paragraph_id} for paragraph_id in candidate
            ]
            repaired_qids.append(qid)

        prediction["summary"] = copy.deepcopy(ZERO_TOKEN_SUMMARY)
        expected_cardinality = self.protocol.get("batch_cardinality", {}).get(split)
        accepted_prediction, batch_accepted = self.refiner.batch_accept(
            previous_prediction=read_json(base_path),
            candidate_prediction=prediction,
            expected_cardinality=(
                int(expected_cardinality) if expected_cardinality is not None else None
            ),
        )
        if not batch_accepted:
            accepted_prediction["summary"] = copy.deepcopy(ZERO_TOKEN_SUMMARY)
        return ReproductionResult(
            split=split,
            prediction=accepted_prediction,
            repaired_qids=tuple(repaired_qids if batch_accepted else ()),
            evidence_count=self.refiner.evidence_cardinality(accepted_prediction),
            batch_accepted=batch_accepted,
        )

    def reproduce_to(self, split: str, output_path: Path) -> ReproductionResult:
        result = self.reproduce(split)
        write_json(output_path, result.prediction)
        return result
