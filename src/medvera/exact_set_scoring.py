from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .evidence_construction import (
    ProtocolAwareEvidenceConstructor,
    RecordIndex,
    encounter_position,
    protocol_family,
)


MODEL_FAMILIES = (
    "logistic_regression",
    "random_forest",
    "histogram_gradient_boosting",
)


def candidate_set_features(
    question: dict[str, Any],
    index: RecordIndex,
    candidate: list[str],
    base_set: list[str],
    constructor: ProtocolAwareEvidenceConstructor,
) -> dict[str, Any]:
    roles = constructor.roles(index, candidate)
    expected_signature = constructor.expected_signature(question)
    compiled = constructor.compile(question, index)
    paragraphs = [index.by_id[paragraph_id] for paragraph_id in candidate]
    ordinals = [
        encounter_position(paragraph_id)[0]
        for paragraph_id in candidate
        if paragraph_id.startswith("encounter_")
    ]
    sections = {paragraph.section_id for paragraph in paragraphs}
    return {
        "family": protocol_family(question),
        "count_match": int(len(candidate) == constructor.expected_count(question)),
        "signature_match": int(roles == expected_signature),
        "missing_roles": max(0, len(expected_signature) - len(roles)),
        "extra_roles": max(0, len(roles) - len(expected_signature)),
        "is_protocol_compiled": int(compiled == candidate),
        "is_base_set": int(base_set == candidate),
        "role_coverage": len(set(roles) & set(expected_signature)),
        "supplemental_count": sum(paragraph.is_supplemental for paragraph in paragraphs),
        "same_encounter": int(len(sections) == 1),
        "cross_encounter": int(len(sections) > 1),
        "min_encounter": min(ordinals) if ordinals else 0,
        "max_encounter": max(ordinals) if ordinals else 0,
        "graph_connected": int(roles == expected_signature),
        "added_count": len(set(candidate) - set(base_set)),
        "removed_count": len(set(base_set) - set(candidate)),
    }


@dataclass(frozen=True)
class CandidateScore:
    candidate: list[str]
    probability: float


class ExactSetScorer:
    """Module 3 complete-set scorer backed by LR, RF, or HGB."""

    def __init__(
        self,
        family: str = "logistic_regression",
        random_state: int = 20260713,
    ) -> None:
        if family not in MODEL_FAMILIES:
            raise ValueError(f"unknown model family: {family}")
        self.family = family
        self.random_state = random_state
        self.vectorizer: Any = None
        self.model: Any = None
        self.constant_probability: float | None = None
        self.dense = family != "logistic_regression"

    def _build_model(self) -> tuple[Any, Any]:
        try:
            from sklearn.ensemble import (
                HistGradientBoostingClassifier,
                RandomForestClassifier,
            )
            from sklearn.feature_extraction import DictVectorizer
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Exact-set training requires: pip install -e '.[train]'"
            ) from exc
        vectorizer = DictVectorizer(sparse=True)
        if self.family == "logistic_regression":
            model = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=self.random_state,
            )
        elif self.family == "random_forest":
            model = RandomForestClassifier(
                n_estimators=80,
                max_depth=7,
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=1,
            )
        else:
            model = HistGradientBoostingClassifier(
                max_iter=80,
                max_depth=5,
                learning_rate=0.08,
                random_state=self.random_state,
            )
        return vectorizer, model

    def fit(
        self,
        feature_rows: list[dict[str, Any]],
        labels: list[int],
    ) -> "ExactSetScorer":
        if len(feature_rows) != len(labels):
            raise ValueError("feature_rows and labels must have equal length")
        if not feature_rows:
            raise ValueError("cannot fit an exact-set scorer without examples")
        unique_labels = set(labels)
        if len(unique_labels) == 1:
            self.constant_probability = float(labels[0])
            return self
        self.vectorizer, self.model = self._build_model()
        matrix = self.vectorizer.fit_transform(feature_rows)
        self.model.fit(matrix.toarray() if self.dense else matrix, labels)
        return self

    def score_features(self, feature_rows: list[dict[str, Any]]) -> list[float]:
        if not feature_rows:
            return []
        if self.constant_probability is not None:
            return [self.constant_probability] * len(feature_rows)
        if self.vectorizer is None or self.model is None:
            raise RuntimeError("ExactSetScorer has not been fitted")
        matrix = self.vectorizer.transform(feature_rows)
        probabilities = self.model.predict_proba(
            matrix.toarray() if self.dense else matrix
        )
        positive_index = list(self.model.classes_).index(1)
        return [float(value) for value in probabilities[:, positive_index]]

    def rank(
        self,
        question: dict[str, Any],
        index: RecordIndex,
        candidates: list[list[str]],
        base_set: list[str],
        constructor: ProtocolAwareEvidenceConstructor,
    ) -> list[CandidateScore]:
        features = [
            candidate_set_features(question, index, candidate, base_set, constructor)
            for candidate in candidates
        ]
        scores = self.score_features(features)
        return sorted(
            (
                CandidateScore(candidate=list(candidate), probability=score)
                for candidate, score in zip(candidates, scores)
            ),
            key=lambda item: (-item.probability, tuple(item.candidate)),
        )


class CurrentSetEditDetector:
    """Separate Module 3 detector deciding whether the current set should change."""

    def __init__(self, random_state: int = 20260713) -> None:
        self.random_state = random_state
        self.vectorizer: Any = None
        self.model: Any = None
        self.constant_probability: float | None = None

    def fit(
        self,
        feature_rows: list[dict[str, Any]],
        labels: list[int],
    ) -> "CurrentSetEditDetector":
        if len(feature_rows) != len(labels):
            raise ValueError("feature_rows and labels must have equal length")
        if not feature_rows:
            raise ValueError("cannot fit an edit detector without examples")
        if len(set(labels)) == 1:
            self.constant_probability = float(labels[0])
            return self
        try:
            from sklearn.feature_extraction import DictVectorizer
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Edit-detector training requires: pip install -e '.[train]'"
            ) from exc
        self.vectorizer = DictVectorizer(sparse=True)
        matrix = self.vectorizer.fit_transform(feature_rows)
        self.model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=self.random_state,
        ).fit(matrix, labels)
        return self

    def probability(self, features: dict[str, Any]) -> float:
        if self.constant_probability is not None:
            return self.constant_probability
        if self.vectorizer is None or self.model is None:
            raise RuntimeError("CurrentSetEditDetector has not been fitted")
        matrix = self.vectorizer.transform([features])
        positive_index = list(self.model.classes_).index(1)
        return float(self.model.predict_proba(matrix)[0, positive_index])


def ensemble_margin_support(margins: Iterable[float]) -> bool:
    """Return the binary support vote used by one fold-seed unit."""

    values = list(margins)
    if len(values) != 3:
        raise ValueError("one fold-seed unit must provide three model margins")
    return sum(values) > 0.0 and sum(value >= 0.0 for value in values) >= 2
