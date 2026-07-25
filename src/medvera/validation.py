from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .evidence_construction import load_record_indexes
from .io import evidence_ids, read_json, read_jsonl, sha256_file


VALID_TRUST_LABELS = {"trusted", "untrusted", "insufficient_evidence"}
TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


@dataclass(frozen=True)
class SchemaReport:
    passed: bool
    qid_count: int
    evidence_count: int
    missing_qids: tuple[str, ...]
    extra_qids: tuple[str, ...]
    duplicate_evidence_ids: int
    invalid: tuple[dict[str, Any], ...]
    token_statistics_valid: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_prediction(
    prediction: dict[str, Any],
    questions_path: Path,
    records_path: Path,
) -> SchemaReport:
    questions = read_jsonl(questions_path)
    indexes = load_record_indexes(records_path)
    expected_qids = {str(question["qid"]) for question in questions}
    actual_qids = set(prediction) - {"summary"}
    invalid: list[dict[str, Any]] = []
    duplicate_count = 0
    evidence_count = 0

    for question in questions:
        qid = str(question["qid"])
        record_id = str(question["record_id"])
        row = prediction.get(qid)
        if not isinstance(row, dict):
            invalid.append({"qid": qid, "reason": "missing prediction object"})
            continue
        if row.get("trust_label") not in VALID_TRUST_LABELS:
            invalid.append({"qid": qid, "reason": "invalid trust_label"})
        if not isinstance(row.get("risk_types"), list):
            invalid.append({"qid": qid, "reason": "risk_types must be a list"})
        if not isinstance(row.get("evidence"), list):
            invalid.append({"qid": qid, "reason": "evidence must be a list"})
            continue
        paragraph_ids = evidence_ids(row)
        evidence_count += len(paragraph_ids)
        duplicate_count += len(paragraph_ids) - len(set(paragraph_ids))
        if len(paragraph_ids) != len(row["evidence"]):
            invalid.append({"qid": qid, "reason": "invalid evidence object"})
        unknown = [
            paragraph_id
            for paragraph_id in paragraph_ids
            if paragraph_id not in indexes[record_id].by_id
        ]
        if unknown:
            invalid.append(
                {
                    "qid": qid,
                    "reason": "paragraph does not belong to associated record",
                    "paragraph_ids": unknown,
                }
            )

    token_payload = prediction.get("summary", {}).get("total_tokens", {})
    token_valid = isinstance(token_payload, dict) and all(
        isinstance(token_payload.get(key), (int, float))
        and token_payload.get(key) == 0
        for key in TOKEN_KEYS
    )
    missing = tuple(sorted(expected_qids - actual_qids))
    extra = tuple(sorted(actual_qids - expected_qids))
    return SchemaReport(
        passed=(
            not missing
            and not extra
            and not invalid
            and duplicate_count == 0
            and token_valid
        ),
        qid_count=len(actual_qids),
        evidence_count=evidence_count,
        missing_qids=missing,
        extra_qids=extra,
        duplicate_evidence_ids=duplicate_count,
        invalid=tuple(invalid),
        token_statistics_valid=token_valid,
    )


def run_dev_evaluation(
    repository_root: Path,
    data_dir: Path,
    prediction_path: Path,
) -> dict[str, Any]:
    evaluator = Path(__file__).with_name("official_evaluate.py")
    gold = data_dir / "dev" / "questions.jsonl"
    with tempfile.TemporaryDirectory(prefix="medvera_eval_") as temporary:
        report_path = Path(temporary) / "report.json"
        environment = dict(os.environ)
        completed = subprocess.run(
            [
                sys.executable,
                str(evaluator),
                "--gold",
                str(gold),
                "--predictions",
                str(prediction_path),
                "--output",
                str(report_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        report = read_json(report_path) if report_path.is_file() else {}
        return {
            "passed": completed.returncode == 0,
            "returncode": completed.returncode,
            "report": report,
            "stderr": completed.stderr,
        }


def artifact_hash_report(repository_root: Path) -> dict[str, Any]:
    manifest_path = repository_root / "artifacts" / "manifest.json"
    manifest = read_json(manifest_path)
    mismatches = []
    for entry in manifest["files"]:
        path = repository_root / entry["path"]
        actual = sha256_file(path) if path.is_file() else None
        if actual != entry["sha256"]:
            mismatches.append(
                {
                    "path": entry["path"],
                    "expected": entry["sha256"],
                    "actual": actual,
                }
            )
    return {
        "passed": not mismatches,
        "checked": len(manifest["files"]),
        "mismatches": mismatches,
    }


def reference_match(
    repository_root: Path,
    split: str,
    prediction_path: Path,
) -> dict[str, Any]:
    if split not in {"dev", "test"}:
        raise ValueError("split must be 'dev' or 'test'")
    manifest = read_json(repository_root / "artifacts" / "manifest.json")
    actual_hash = sha256_file(prediction_path)
    expected_hash = str(manifest["expected_output_sha256"][split])
    return {
        "passed": actual_hash == expected_hash,
        "actual_sha256": actual_hash,
        "expected_sha256": expected_hash,
    }


def print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))
