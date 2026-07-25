from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


PUBLIC_QUERY_FIELDS = (
    "qid",
    "record_id",
    "question",
    "claim",
    "proposed_response",
    "options",
    "answer_format",
)

ZERO_TOKEN_SUMMARY = {
    "total_tokens": {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def public_view(question: dict[str, Any]) -> dict[str, Any]:
    return {field: question.get(field) for field in PUBLIC_QUERY_FIELDS if field in question}


def evidence_ids(row: dict[str, Any]) -> list[str]:
    evidence = row.get("evidence", [])
    if not isinstance(evidence, list):
        return []
    return [
        str(item["paragraph_id"])
        for item in evidence
        if isinstance(item, dict) and item.get("paragraph_id")
    ]


def prediction_field_hash(prediction: dict[str, Any], field: str) -> str:
    return canonical_hash(
        {
            qid: row.get(field)
            for qid, row in prediction.items()
            if qid != "summary" and isinstance(row, dict)
        }
    )
