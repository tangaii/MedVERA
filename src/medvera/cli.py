from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from .evidence_construction import load_record_indexes
from .io import read_jsonl, sha256_file, write_json
from .pipeline import MedVERAPipeline
from .training import TrainableMedVERA, TrainingConfig
from .validation import (
    artifact_hash_report,
    reference_match,
    run_dev_evaluation,
    validate_prediction,
)


def default_repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_dir(repository_root: Path) -> Path:
    configured = os.environ.get("MEDVERA_DATA_DIR")
    return Path(configured).expanduser() if configured else repository_root / "data" / "MedLongTrust-EHR"


def _common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_repository_root(),
        help="MedVERA repository root",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="official MedLongTrust-EHR release root",
    )
    return parser


def _resolve_data_dir(args: argparse.Namespace) -> Path:
    return (args.data_dir or default_data_dir(args.repo_root)).resolve()


def reproduce_main(argv: list[str] | None = None) -> int:
    parser = _common_parser("Reproduce MedVERA development and competition predictions.")
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--verify-hash", action="store_true")
    args = parser.parse_args(argv)
    repository_root = args.repo_root.resolve()
    data_dir = _resolve_data_dir(args)
    output_dir = args.output_dir.resolve()
    pipeline = MedVERAPipeline(repository_root, data_dir)
    splits = ("dev", "test") if args.split == "all" else (args.split,)
    payload: dict[str, Any] = {
        "repository_root": str(repository_root),
        "data_dir": str(data_dir),
        "outputs": {},
    }
    passed = True
    for split in splits:
        filename = "dev_predictions.json" if split == "dev" else "submission.json"
        output_path = output_dir / filename
        result = pipeline.reproduce_to(split, output_path)
        split_payload: dict[str, Any] = {
            "path": str(output_path),
            "repaired_qids": list(result.repaired_qids),
            "evidence_count": result.evidence_count,
            "batch_accepted": result.batch_accepted,
        }
        if args.verify_hash:
            split_payload["reference"] = reference_match(
                repository_root,
                split,
                output_path,
            )
            passed = passed and split_payload["reference"]["passed"]
        payload["outputs"][split] = split_payload
        passed = passed and result.batch_accepted
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if passed else 1


def verify_main(argv: list[str] | None = None) -> int:
    parser = _common_parser("Verify MedVERA artifacts and exact frozen reproduction.")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    repository_root = args.repo_root.resolve()
    data_dir = _resolve_data_dir(args)
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.output_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="medvera_verify_")
        output_dir = Path(temporary.name)
    else:
        output_dir = args.output_dir.resolve()
    try:
        pipeline = MedVERAPipeline(repository_root, data_dir)
        checks: dict[str, Any] = {"artifacts": artifact_hash_report(repository_root)}
        for split, filename in (
            ("dev", "dev_predictions.json"),
            ("test", "submission.json"),
        ):
            output_path = output_dir / filename
            result = pipeline.reproduce_to(split, output_path)
            schema = validate_prediction(
                result.prediction,
                data_dir / split / "questions.jsonl",
                data_dir / split / "records.jsonl",
            )
            checks[split] = {
                "schema": schema.to_dict(),
                "reference": reference_match(repository_root, split, output_path),
                "batch_accepted": result.batch_accepted,
                "repaired_qids": list(result.repaired_qids),
            }
        checks["dev"]["evaluation"] = run_dev_evaluation(
            repository_root,
            data_dir,
            output_dir / "dev_predictions.json",
        )
        dev_report = checks["dev"]["evaluation"].get("report", {})
        exact_dev_score = (
            dev_report.get("final_score") == 100.0
            and dev_report.get("trust_label_accuracy") == 1.0
            and dev_report.get("risk_type", {}).get("micro_f1") == 1.0
            and dev_report.get("evidence", {}).get("f1") == 1.0
        )
        checks["dev"]["score_100"] = exact_dev_score
        passed = (
            checks["artifacts"]["passed"]
            and checks["dev"]["schema"]["passed"]
            and checks["test"]["schema"]["passed"]
            and checks["dev"]["reference"]["passed"]
            and checks["test"]["reference"]["passed"]
            and checks["dev"]["batch_accepted"]
            and checks["test"]["batch_accepted"]
            and checks["dev"]["evaluation"]["passed"]
            and exact_dev_score
        )
        print(json.dumps({"passed": passed, "checks": checks}, ensure_ascii=False, indent=2))
        return 0 if passed else 1
    finally:
        if temporary is not None:
            temporary.cleanup()


def _load_split(
    data_dir: Path,
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split_dir = data_dir / split
    questions_path = split_dir / "questions.jsonl"
    records_path = split_dir / "records.jsonl"
    if not questions_path.is_file() or not records_path.is_file():
        raise FileNotFoundError(
            f"{split_dir} must contain questions.jsonl and records.jsonl"
        )
    return read_jsonl(questions_path), load_record_indexes(records_path)


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one integer seed is required")
    return seeds


def train_main(argv: list[str] | None = None) -> int:
    parser = _common_parser("Train MedVERA from the official raw training split.")
    parser.add_argument("--model-dir", type=Path, default=Path("models/medvera"))
    parser.add_argument("--random-state", type=int, default=20260713)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=(20260713, 20260720, 20260727),
        help="comma-separated random seeds",
    )
    parser.add_argument(
        "--consensus-min-support",
        type=int,
        default=None,
        help="support votes required; defaults to 17/18 or all units for a custom ensemble",
    )
    parser.add_argument(
        "--evaluate-dev",
        action="store_true",
        help="predict and evaluate dev after fitting; dev is never used for training",
    )
    parser.add_argument(
        "--paragraph-candidate-limit",
        type=int,
        default=64,
    )
    args = parser.parse_args(argv)
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if args.paragraph_candidate_limit < 1:
        parser.error("--paragraph-candidate-limit must be positive")
    unit_count = len(args.seeds) * args.folds
    consensus_min_support = (
        args.consensus_min_support
        if args.consensus_min_support is not None
        else (17 if unit_count == 18 else unit_count)
    )
    if not 1 <= consensus_min_support <= unit_count:
        parser.error(
            f"--consensus-min-support must be between 1 and {unit_count}"
        )

    repository_root = args.repo_root.resolve()
    data_dir = _resolve_data_dir(args)
    model_dir = args.model_dir.resolve()
    train_questions, train_indexes = _load_split(data_dir, "train")
    model = TrainableMedVERA(
        TrainingConfig(
            random_state=args.random_state,
            seeds=args.seeds,
            folds=args.folds,
            consensus_min_support=consensus_min_support,
            paragraph_candidate_limit=args.paragraph_candidate_limit,
        )
    ).fit(train_questions, train_indexes)

    dev_payload: dict[str, Any] | None = None
    if args.evaluate_dev:
        dev_questions, dev_indexes = _load_split(data_dir, "dev")
        prediction = model.predict(dev_questions, dev_indexes)
        dev_path = model_dir / "dev_predictions.json"
        write_json(dev_path, prediction)
        schema = validate_prediction(
            prediction,
            data_dir / "dev" / "questions.jsonl",
            data_dir / "dev" / "records.jsonl",
        )
        evaluation = run_dev_evaluation(
            repository_root,
            data_dir,
            dev_path,
        )
        dev_payload = {
            "prediction_path": str(dev_path),
            "schema": schema.to_dict(),
            "evaluation": evaluation,
        }
        model.training_report["held_out_dev"] = dev_payload

    model_path = model.save(model_dir)
    payload = {
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "training_report": model.training_report,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if dev_payload is None:
        return 0
    return 0 if (
        dev_payload["schema"]["passed"]
        and dev_payload["evaluation"]["passed"]
    ) else 1


def predict_main(argv: list[str] | None = None) -> int:
    parser = _common_parser("Run a model trained by medvera-train.")
    parser.add_argument("--model-dir", type=Path, default=Path("models/medvera"))
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="run the official evaluator; valid only for dev",
    )
    args = parser.parse_args(argv)
    if args.evaluate and args.split != "dev":
        parser.error("--evaluate is only available for the dev split")
    repository_root = args.repo_root.resolve()
    data_dir = _resolve_data_dir(args)
    model = TrainableMedVERA.load(args.model_dir.resolve())
    questions, indexes = _load_split(data_dir, args.split)
    prediction = model.predict(questions, indexes)
    default_name = "dev_predictions.json" if args.split == "dev" else "submission.json"
    output_path = (args.output or Path("outputs") / default_name).resolve()
    write_json(output_path, prediction)
    schema = validate_prediction(
        prediction,
        data_dir / args.split / "questions.jsonl",
        data_dir / args.split / "records.jsonl",
    )
    payload: dict[str, Any] = {
        "output": str(output_path),
        "sha256": sha256_file(output_path),
        "schema": schema.to_dict(),
    }
    if args.evaluate:
        payload["evaluation"] = run_dev_evaluation(
            repository_root,
            data_dir,
            output_path,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if schema.passed else 1


if __name__ == "__main__":  # pragma: no cover
    command = Path(sys.argv[0]).stem
    if "train" in command:
        raise SystemExit(train_main())
    if "predict" in command:
        raise SystemExit(predict_main())
    raise SystemExit(verify_main() if "verify" in command else reproduce_main())
