from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .evidence_construction import (
    HybridParagraphRetriever,
    ProtocolAwareEvidenceConstructor,
    RecordIndex,
    canonical_set,
    encounter_position,
    extract_dates,
    extract_numbers,
    extract_units,
    paragraph_role,
    protocol_family,
    tokenize,
)
from .exact_set_scoring import (
    MODEL_FAMILIES,
    CurrentSetEditDetector,
    ExactSetScorer,
    candidate_set_features,
    ensemble_margin_support,
)
from .io import ZERO_TOKEN_SUMMARY, evidence_ids, public_view, write_json
from .semantic_assessment import (
    SemanticPrediction,
    SemanticTrustRiskAssessor,
    StructuredSemanticCalibrator,
    normalize_text,
)


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration persisted with a trainable MedVERA model."""

    random_state: int = 20260713
    seeds: tuple[int, ...] = (20260713, 20260720, 20260727)
    folds: int = 6
    consensus_min_support: int = 17
    paragraph_candidate_limit: int = 64

    @property
    def unit_count(self) -> int:
        return len(self.seeds) * self.folds


@dataclass
class ParagraphCandidate:
    paragraph_id: str
    retrieval_score: float
    sources: tuple[str, ...]


class ParagraphEvidenceRanker:
    """Supervised paragraph scorer used as a protocol fallback and research hook."""

    def __init__(self, random_state: int = 20260713) -> None:
        self.random_state = random_state
        self.vectorizer: Any = None
        self.model: Any = None
        self.constant_probability: float | None = None
        self.training_summary: dict[str, Any] = {}

    @staticmethod
    def _features(
        question: dict[str, Any],
        index: RecordIndex,
        candidate: ParagraphCandidate,
        prediction: SemanticPrediction,
        expected_roles: tuple[str, ...],
    ) -> dict[str, Any]:
        paragraph = index.by_id[candidate.paragraph_id]
        question_tokens = set(tokenize(question.get("question")))
        claim_tokens = set(tokenize(question.get("claim")))
        response_tokens = set(tokenize(question.get("proposed_response")))
        options_tokens = set(
            tokenize(json.dumps(question.get("options", {}), ensure_ascii=False))
        )
        query_text = " ".join(
            normalize_text(question.get(field))
            for field in ("question", "claim", "proposed_response")
        )
        role = paragraph_role(paragraph)
        features: dict[str, Any] = {
            "family": protocol_family(question),
            "subtype": prediction.subtype,
            "predicted_trust": prediction.trust_label,
            "paragraph_role": role,
            "retrieval_score": candidate.retrieval_score,
            "question_overlap": len(question_tokens & set(paragraph.tokens)),
            "claim_overlap": len(claim_tokens & set(paragraph.tokens)),
            "response_overlap": len(response_tokens & set(paragraph.tokens)),
            "options_overlap": len(options_tokens & set(paragraph.tokens)),
            "number_overlap": len(extract_numbers(query_text) & set(paragraph.numbers)),
            "date_overlap": len(extract_dates(query_text) & set(paragraph.dates)),
            "unit_overlap": len(extract_units(query_text) & set(paragraph.units)),
            "expected_role": int(role in expected_roles),
            "is_context": int(paragraph.is_encounter_context),
            "is_overview": int(paragraph.is_patient_overview),
            "is_supplemental": int(paragraph.is_supplemental),
            "paragraph_order": paragraph.order,
            "paragraph_length": sum(paragraph.token_counts.values()),
            "source_count": len(candidate.sources),
        }
        for source in candidate.sources:
            features[f"source={source}"] = 1
        for risk in prediction.risk_types:
            features[f"risk={risk}"] = 1
        return features

    @staticmethod
    def _dependencies() -> tuple[Any, Any, Any]:
        try:
            from sklearn.feature_extraction import DictVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Training requires the optional dependencies: pip install -e '.[train]'"
            ) from exc
        return DictVectorizer, LogisticRegression, roc_auc_score

    def fit(
        self,
        questions: list[dict[str, Any]],
        indexes: dict[str, RecordIndex],
        predictions: dict[str, SemanticPrediction],
        candidate_builder: Any,
        constructor: ProtocolAwareEvidenceConstructor,
    ) -> "ParagraphEvidenceRanker":
        DictVectorizer, LogisticRegression, roc_auc_score = self._dependencies()
        rows: list[dict[str, Any]] = []
        labels: list[int] = []
        for question in questions:
            qid = str(question["qid"])
            index = indexes[str(question["record_id"])]
            gold = set(gold_evidence_ids(question))
            candidates = candidate_builder(question, index, gold)
            expected_roles = constructor.expected_signature(question)
            for candidate in candidates.values():
                rows.append(
                    self._features(
                        question,
                        index,
                        candidate,
                        predictions[qid],
                        expected_roles,
                    )
                )
                labels.append(int(candidate.paragraph_id in gold))
        if not rows:
            raise ValueError("no paragraph candidates were generated for training")
        if len(set(labels)) == 1:
            self.constant_probability = float(labels[0])
            self.training_summary = {
                "samples": len(labels),
                "positives": sum(labels),
                "negatives": len(labels) - sum(labels),
            }
            return self
        self.vectorizer = DictVectorizer(sparse=True)
        matrix = self.vectorizer.fit_transform(rows)
        self.model = LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            C=2.0,
            random_state=self.random_state,
        ).fit(matrix, labels)
        probabilities = self.model.predict_proba(matrix)[:, list(self.model.classes_).index(1)]
        self.training_summary = {
            "samples": len(labels),
            "positives": int(sum(labels)),
            "negatives": int(len(labels) - sum(labels)),
            "train_auc": float(roc_auc_score(labels, probabilities)),
        }
        return self

    def score(
        self,
        question: dict[str, Any],
        index: RecordIndex,
        candidates: dict[str, ParagraphCandidate],
        prediction: SemanticPrediction,
        expected_roles: tuple[str, ...],
    ) -> dict[str, float]:
        if not candidates:
            return {}
        if self.constant_probability is not None:
            return {
                paragraph_id: self.constant_probability
                for paragraph_id in candidates
            }
        if self.vectorizer is None or self.model is None:
            raise RuntimeError("ParagraphEvidenceRanker has not been fitted")
        ordered = list(candidates.values())
        rows = [
            self._features(question, index, candidate, prediction, expected_roles)
            for candidate in ordered
        ]
        matrix = self.vectorizer.transform(rows)
        positive_index = list(self.model.classes_).index(1)
        probabilities = self.model.predict_proba(matrix)[:, positive_index]
        return {
            candidate.paragraph_id: float(probability)
            for candidate, probability in zip(ordered, probabilities)
        }


@dataclass
class SetScoringUnit:
    seed: int
    fold: int
    models: dict[str, ExactSetScorer]


def gold_evidence_ids(question: dict[str, Any]) -> list[str]:
    target = question.get("target")
    if isinstance(target, dict):
        values = evidence_ids(target)
        if values or target.get("evidence") == []:
            return values
    return evidence_ids(question)


def record_grouped_fold(record_id: str, seed: int, folds: int) -> int:
    digest = hashlib.sha256(f"{seed}|{record_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def evidence_metrics(
    predicted: dict[str, list[str]],
    questions: Iterable[dict[str, Any]],
) -> dict[str, float | int]:
    true_positive = false_positive = false_negative = exact = total = 0
    for question in questions:
        qid = str(question["qid"])
        prediction = set(predicted[qid])
        gold = set(gold_evidence_ids(question))
        true_positive += len(prediction & gold)
        false_positive += len(prediction - gold)
        false_negative += len(gold - prediction)
        exact += prediction == gold
        total += 1
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "exact_set_accuracy": exact / total if total else 1.0,
        "questions": total,
    }


class TrainableMedVERA:
    """Raw-data train/save/load/predict pipeline kept separate from frozen reproduction."""

    model_filename = "model.joblib"
    metadata_filename = "training_report.json"

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()
        self.semantic_assessor = SemanticTrustRiskAssessor(
            random_state=self.config.random_state
        )
        self.semantic_calibrator = StructuredSemanticCalibrator(
            random_state=self.config.random_state
        )
        self.retriever = HybridParagraphRetriever(
            lexical_top_k=16,
            candidate_limit=self.config.paragraph_candidate_limit,
        )
        self.constructor = ProtocolAwareEvidenceConstructor()
        self.paragraph_ranker = ParagraphEvidenceRanker(
            random_state=self.config.random_state
        )
        self.edit_detector = CurrentSetEditDetector(
            random_state=self.config.random_state
        )
        self.set_units: list[SetScoringUnit] = []
        self.training_report: dict[str, Any] = {}

    def _semantic_predictions(
        self,
        questions: list[dict[str, Any]],
    ) -> dict[str, SemanticPrediction]:
        base = self.semantic_assessor.predict(questions)
        return {
            str(question["qid"]): self.semantic_calibrator.predict(
                question,
                base[str(question["qid"])],
            )
            for question in questions
        }

    def _paragraph_candidates(
        self,
        question: dict[str, Any],
        index: RecordIndex,
        required_ids: set[str] | None = None,
    ) -> dict[str, ParagraphCandidate]:
        result: dict[str, ParagraphCandidate] = {}

        def add(paragraph_id: str, score: float, source: str) -> None:
            if paragraph_id not in index.by_id:
                return
            current = result.get(paragraph_id)
            if current is None:
                result[paragraph_id] = ParagraphCandidate(
                    paragraph_id=paragraph_id,
                    retrieval_score=score,
                    sources=(source,),
                )
                return
            result[paragraph_id] = ParagraphCandidate(
                paragraph_id=paragraph_id,
                retrieval_score=max(score, current.retrieval_score),
                sources=tuple(sorted(set(current.sources) | {source})),
            )

        for candidate in self.retriever.retrieve(question, index):
            add(candidate.paragraph_id, candidate.score, "|".join(candidate.sources))
        expected_roles = set(self.constructor.expected_signature(question))
        for paragraph in index.paragraphs:
            if (
                paragraph_role(paragraph) in expected_roles
                and not paragraph.is_supplemental
            ):
                add(paragraph.paragraph_id, 0.0, "role_pool")
        compiled = self.constructor.compile(question, index)
        for paragraph_id in compiled or []:
            add(paragraph_id, 100.0, "protocol")
        for paragraph_id in required_ids or set():
            add(paragraph_id, 100.0, "supervision")
        ordered = sorted(
            result.values(),
            key=lambda item: (-item.retrieval_score, item.paragraph_id),
        )
        return {
            item.paragraph_id: item
            for item in ordered[: self.config.paragraph_candidate_limit]
        }

    @staticmethod
    def _same_section_pair(
        index: RecordIndex,
        main_id: str,
    ) -> list[str] | None:
        context_id = index.context_by_section.get(index.by_id[main_id].section_id)
        result = canonical_set((context_id, main_id))
        return result if len(result) == 2 else None

    def _ranker_fallback(
        self,
        question: dict[str, Any],
        index: RecordIndex,
        prediction: SemanticPrediction,
    ) -> list[str]:
        signature = self.constructor.expected_signature(question)
        if not signature:
            return []
        candidates = self._paragraph_candidates(question, index)
        scores = self.paragraph_ranker.score(
            question,
            index,
            candidates,
            prediction,
            signature,
        )
        family = protocol_family(question)
        if family in {"condition", "medication", "allergy"}:
            wanted_role = {
                "condition": "condition_main",
                "medication": "medication_main",
                "allergy": "allergy",
            }[family]
            main_ids = [
                paragraph_id
                for paragraph_id in candidates
                if paragraph_role(index.by_id[paragraph_id]) == wanted_role
            ]
            if main_ids:
                best = max(main_ids, key=lambda pid: (scores.get(pid, 0.0), pid))
                return self._same_section_pair(index, best) or [best]
        selected: list[str] = []
        for role in signature:
            choices = [
                paragraph_id
                for paragraph_id in candidates
                if paragraph_id not in selected
                and paragraph_role(index.by_id[paragraph_id]) == role
            ]
            if choices:
                selected.append(max(choices, key=lambda pid: (scores.get(pid, 0.0), pid)))
        return canonical_set(selected)

    def _trusted_anchors(
        self,
        questions: list[dict[str, Any]],
        indexes: dict[str, RecordIndex],
        predictions: dict[str, SemanticPrediction],
    ) -> dict[tuple[str, str], list[str]]:
        anchors: dict[tuple[str, str], list[str]] = {}
        ordered = sorted(
            questions,
            key=lambda row: (
                bool(normalize_text(row.get("proposed_response"))),
                str(row["qid"]),
            ),
        )
        for question in ordered:
            family = protocol_family(question)
            qid = str(question["qid"])
            if family not in {"condition", "medication"}:
                continue
            if predictions[qid].trust_label != "trusted":
                continue
            index = indexes[str(question["record_id"])]
            compiled = self.constructor.compile(question, index)
            if compiled is not None:
                anchors.setdefault((str(question["record_id"]), family), compiled)
        return anchors

    def _base_evidence(
        self,
        question: dict[str, Any],
        index: RecordIndex,
        prediction: SemanticPrediction,
        anchors: dict[tuple[str, str], list[str]],
    ) -> list[str]:
        family = protocol_family(question)
        anchor = anchors.get((str(question["record_id"]), family))
        if family in {"condition", "medication"} and anchor is not None:
            return list(anchor)
        compiled = self.constructor.compile(question, index)
        if compiled is not None:
            return list(compiled)
        return self._ranker_fallback(question, index, prediction)

    def _candidate_sets(
        self,
        question: dict[str, Any],
        index: RecordIndex,
        prediction: SemanticPrediction,
        base: list[str],
        anchors: dict[tuple[str, str], list[str]],
        include_gold: bool,
    ) -> list[list[str]]:
        candidates: list[list[str]] = [canonical_set(base)]
        compiled = self.constructor.compile(question, index)
        if compiled is not None:
            candidates.append(canonical_set(compiled))
        anchor = anchors.get((str(question["record_id"]), protocol_family(question)))
        if anchor is not None:
            candidates.append(canonical_set(anchor))
        fallback = self._ranker_fallback(question, index, prediction)
        if fallback:
            candidates.append(canonical_set(fallback))
        if include_gold:
            gold = gold_evidence_ids(question)
            candidates.append(canonical_set(gold))
            if gold:
                candidates.append(canonical_set(gold[:-1]))
                for position, paragraph_id in enumerate(gold):
                    role = paragraph_role(index.by_id[paragraph_id])
                    alternatives = sorted(
                        (
                            paragraph.paragraph_id
                            for paragraph in index.paragraphs
                            if paragraph.paragraph_id not in gold
                            and paragraph_role(paragraph) == role
                            and not paragraph.is_supplemental
                        ),
                        key=encounter_position,
                    )
                    if alternatives:
                        corrupted = list(gold)
                        corrupted[position] = alternatives[0]
                        candidates.append(canonical_set(corrupted))
                        break
        unique: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        for candidate in candidates:
            valid = canonical_set(
                paragraph_id
                for paragraph_id in candidate
                if paragraph_id in index.by_id
            )
            key = tuple(valid)
            if key not in seen:
                seen.add(key)
                unique.append(valid)
        return unique

    def _paragraph_scores(
        self,
        question: dict[str, Any],
        index: RecordIndex,
        prediction: SemanticPrediction,
    ) -> dict[str, float]:
        candidates = self._paragraph_candidates(question, index)
        return self.paragraph_ranker.score(
            question,
            index,
            candidates,
            prediction,
            self.constructor.expected_signature(question),
        )

    def _set_features(
        self,
        question: dict[str, Any],
        index: RecordIndex,
        candidate: list[str],
        base: list[str],
        prediction: SemanticPrediction,
        paragraph_scores: dict[str, float],
    ) -> dict[str, Any]:
        features = candidate_set_features(
            question,
            index,
            candidate,
            base,
            self.constructor,
        )
        values = [paragraph_scores.get(paragraph_id, 0.0) for paragraph_id in candidate]
        features.update(
            {
                "predicted_trust": prediction.trust_label,
                "predicted_subtype": prediction.subtype,
                "paragraph_score_mean": sum(values) / len(values) if values else 0.0,
                "paragraph_score_min": min(values) if values else 0.0,
                "paragraph_score_max": max(values) if values else 0.0,
            }
        )
        return features

    def _edit_features(
        self,
        question: dict[str, Any],
        index: RecordIndex,
        base: list[str],
        prediction: SemanticPrediction,
        paragraph_scores: dict[str, float],
    ) -> dict[str, Any]:
        features = self._set_features(
            question,
            index,
            base,
            base,
            prediction,
            paragraph_scores,
        )
        compiled = self.constructor.compile(question, index)
        features["compiled_available"] = int(compiled is not None)
        features["compiled_differs"] = int(
            compiled is not None and set(compiled) != set(base)
        )
        return features

    def fit(
        self,
        questions: list[dict[str, Any]],
        indexes: dict[str, RecordIndex],
    ) -> "TrainableMedVERA":
        if not questions:
            raise ValueError("training questions are empty")
        missing = {
            str(question["record_id"])
            for question in questions
            if str(question["record_id"]) not in indexes
        }
        if missing:
            raise ValueError(f"missing training records: {sorted(missing)[:5]}")

        self.semantic_assessor.fit(questions)
        self.semantic_calibrator.fit(questions)
        predictions = self._semantic_predictions(questions)
        self.paragraph_ranker.fit(
            questions,
            indexes,
            predictions,
            self._paragraph_candidates,
            self.constructor,
        )
        anchors = self._trusted_anchors(questions, indexes, predictions)

        bases: dict[str, list[str]] = {}
        set_rows: list[dict[str, Any]] = []
        edit_rows: list[dict[str, Any]] = []
        edit_labels: list[int] = []
        for question in questions:
            qid = str(question["qid"])
            index = indexes[str(question["record_id"])]
            prediction = predictions[qid]
            base = self._base_evidence(question, index, prediction, anchors)
            bases[qid] = base
            paragraph_scores = self._paragraph_scores(question, index, prediction)
            gold = set(gold_evidence_ids(question))
            candidate_sets = self._candidate_sets(
                question,
                index,
                prediction,
                base,
                anchors,
                include_gold=True,
            )
            for candidate in candidate_sets:
                set_rows.append(
                    {
                        "record_id": str(question["record_id"]),
                        "features": self._set_features(
                            question,
                            index,
                            candidate,
                            base,
                            prediction,
                            paragraph_scores,
                        ),
                        "label": int(set(candidate) == gold),
                    }
                )
                edit_rows.append(
                    self._edit_features(
                        question,
                        index,
                        candidate,
                        prediction,
                        paragraph_scores,
                    )
                )
                edit_labels.append(int(set(candidate) != gold))
        self.edit_detector.fit(edit_rows, edit_labels)

        self.set_units = []
        for seed in self.config.seeds:
            for fold in range(self.config.folds):
                fit_rows = [
                    row
                    for row in set_rows
                    if record_grouped_fold(
                        row["record_id"],
                        seed,
                        self.config.folds,
                    )
                    != fold
                ]
                if not fit_rows:
                    fit_rows = set_rows
                models = {
                    family: ExactSetScorer(
                        family=family,
                        random_state=seed + fold,
                    ).fit(
                        [row["features"] for row in fit_rows],
                        [row["label"] for row in fit_rows],
                    )
                    for family in MODEL_FAMILIES
                }
                self.set_units.append(SetScoringUnit(seed, fold, models))

        semantic_correct = sum(
            predictions[str(question["qid"])].trust_label
            == str(question["target"]["trust_label"])
            for question in questions
        )
        risk_correct = sum(
            set(predictions[str(question["qid"])].risk_types)
            == set(question["target"].get("risk_types", []))
            for question in questions
        )
        self.training_report = {
            "format_version": 1,
            "training_questions": len(questions),
            "training_records": len(indexes),
            "config": asdict(self.config),
            "semantic": {
                "trust_accuracy": semantic_correct / len(questions),
                "risk_exact_accuracy": risk_correct / len(questions),
            },
            "paragraph_ranker": self.paragraph_ranker.training_summary,
            "base_evidence": evidence_metrics(bases, questions),
            "set_candidate_rows": len(set_rows),
            "set_scoring_units": len(self.set_units),
            "edit_training_sets": len(edit_labels),
            "edit_positive_sets": int(sum(edit_labels)),
        }
        return self

    def _refine_set(
        self,
        question: dict[str, Any],
        index: RecordIndex,
        prediction: SemanticPrediction,
        base: list[str],
        candidates: list[list[str]],
        paragraph_scores: dict[str, float],
    ) -> list[str]:
        if len(candidates) < 2 or not self.set_units:
            return base
        feature_rows = [
            self._set_features(
                question,
                index,
                candidate,
                base,
                prediction,
                paragraph_scores,
            )
            for candidate in candidates
        ]
        base_key = tuple(base)
        support: Counter[tuple[str, ...]] = Counter()
        mean_margin: Counter[tuple[str, ...]] = Counter()
        for unit in self.set_units:
            family_scores = {
                family: model.score_features(feature_rows)
                for family, model in unit.models.items()
            }
            base_position = next(
                (position for position, candidate in enumerate(candidates) if tuple(candidate) == base_key),
                0,
            )
            averages = [
                sum(scores[position] for scores in family_scores.values())
                / len(family_scores)
                for position in range(len(candidates))
            ]
            winner = max(
                range(len(candidates)),
                key=lambda position: (averages[position], tuple(candidates[position])),
            )
            if winner == base_position:
                continue
            margins = [
                scores[winner] - scores[base_position]
                for scores in family_scores.values()
            ]
            if ensemble_margin_support(margins):
                key = tuple(candidates[winner])
                support[key] += 1
                mean_margin[key] += sum(margins) / len(margins)
        if not support:
            return base
        candidate_key = max(
            support,
            key=lambda key: (support[key], mean_margin[key], key),
        )
        required = min(self.config.consensus_min_support, len(self.set_units))
        edit_probability = self.edit_detector.probability(
            self._edit_features(
                question,
                index,
                base,
                prediction,
                paragraph_scores,
            )
        )
        if support[candidate_key] >= required and edit_probability >= 0.5:
            return list(candidate_key)
        return base

    def predict(
        self,
        questions: list[dict[str, Any]],
        indexes: dict[str, RecordIndex],
    ) -> dict[str, Any]:
        questions = [public_view(question) for question in questions]
        predictions = self._semantic_predictions(questions)
        anchors = self._trusted_anchors(questions, indexes, predictions)
        output: dict[str, Any] = {}
        for question in questions:
            qid = str(question["qid"])
            index = indexes[str(question["record_id"])]
            semantic = predictions[qid]
            base = self._base_evidence(question, index, semantic, anchors)
            paragraph_scores = self._paragraph_scores(question, index, semantic)
            candidates = self._candidate_sets(
                question,
                index,
                semantic,
                base,
                anchors,
                include_gold=False,
            )
            evidence = self._refine_set(
                question,
                index,
                semantic,
                base,
                candidates,
                paragraph_scores,
            )
            output[qid] = {
                "trust_label": semantic.trust_label,
                "risk_types": list(semantic.risk_types),
                "evidence": [
                    {"paragraph_id": paragraph_id}
                    for paragraph_id in evidence
                ],
            }
        output["summary"] = json.loads(json.dumps(ZERO_TOKEN_SUMMARY))
        return output

    def save(self, model_dir: Path) -> Path:
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Saving a model requires joblib") from exc
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / self.model_filename
        joblib.dump(self, model_path, compress=3)
        write_json(model_dir / self.metadata_filename, self.training_report)
        return model_path

    @classmethod
    def load(cls, model_dir: Path) -> "TrainableMedVERA":
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Loading a model requires joblib") from exc
        model_path = model_dir / cls.model_filename
        model = joblib.load(model_path)
        if not isinstance(model, cls):
            raise TypeError(f"{model_path} does not contain a {cls.__name__}")
        return model
