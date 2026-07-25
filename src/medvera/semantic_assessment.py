from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import public_view


TRUST_LABELS = ("trusted", "untrusted", "insufficient_evidence")
RISK_TYPES = (
    "hallucination.fact_mismatch",
    "hallucination.fabricated_medication",
    "hallucination.trend_reversal",
    "hallucination.temporal_mismatch",
    "hallucination.indicator_mismatch",
    "privacy.identifier_disclosure",
    "safety.allergy_conflict",
    "insufficient_evidence",
)


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_key(value: Any) -> str:
    text = normalize_text(value).lower().replace("–", "-")
    return normalize_text(re.sub(r"[^a-z0-9%/{}\[\].:+-]+", " ", text))


def infer_query_family(question: dict[str, Any]) -> str:
    question_text = normalize_key(question.get("question"))
    claim = normalize_key(question.get("claim"))
    response = normalize_key(question.get("proposed_response"))
    if "privacy requirements" in question_text or any(
        token in claim or token in response
        for token in (
            "synthetic patient identifier",
            "record identifiers",
            "encounter id",
            "address",
            "birthdate",
            "identifier",
        )
    ):
        return "privacy"
    if "allergy" in claim or "allergy" in response:
        return "allergy"
    if (
        "serum lactate" in claim
        or "insufficiently evidenced" in response
        or "sufficient to determine whether" in claim
        or "not sufficient to determine whether" in claim
    ):
        return "not_enough"
    if (
        "medication timeline" in claim
        or "medication history" in claim
        or "patient received" in claim
        or ("includes" in claim and "medication" in claim)
    ):
        return "medication"
    if (
        "encounter-level observations" in claim
        or "changed from" in claim
        or "increased from" in claim
        or "decreased from" in claim
        or claim.startswith("in encounter ")
    ):
        return "observation"
    return "condition"


def infer_observation_subtype(question: dict[str, Any]) -> str:
    claim = normalize_key(question.get("claim"))
    response = normalize_key(question.get("proposed_response"))
    if "the values for" in response and "reported as if they were" in response:
        return "indicator_mismatch"
    if claim.startswith("in encounter ") and (
        "earlier encounter" in response or "later encounter" in response
    ):
        return "temporal_mismatch"
    if any(
        term in claim
        for term in (
            "increased from",
            "decreased from",
            "rose from",
            "fell from",
            "changed from",
        )
    ):
        return "numeric_trend"
    return "unsupported_observation"


def public_query_text(question: dict[str, Any]) -> str:
    view = public_view(question)
    family = infer_query_family(view)
    subtype = infer_observation_subtype(view) if family == "observation" else "none"
    indicators = (
        f"family={family}",
        f"subtype={subtype}",
        f"has_privacy={int(family == 'privacy')}",
        f"has_allergy={int(family == 'allergy')}",
        f"has_temporal={int(subtype == 'temporal_mismatch')}",
        f"has_indicator_mismatch={int(subtype == 'indicator_mismatch')}",
    )
    fields = (
        normalize_text(view.get("question")),
        normalize_text(view.get("claim")),
        normalize_text(view.get("proposed_response")),
        json.dumps(view.get("options", {}), ensure_ascii=False, sort_keys=True),
    )
    return " ".join(indicators + tuple(field for field in fields if field))


def target_from_question(question: dict[str, Any]) -> tuple[str, list[str]]:
    target = question.get("target", {})
    if not isinstance(target, dict):
        raise ValueError("training question is missing target annotations")
    trust = str(target.get("trust_label", ""))
    risks = [str(value) for value in target.get("risk_types", [])]
    if trust not in TRUST_LABELS:
        raise ValueError(f"invalid trust label in training data: {trust!r}")
    return trust, risks


def numeric_direction_relation(question: dict[str, Any]) -> str:
    """Encode whether an asserted numeric direction agrees with its endpoints."""

    claim = normalize_text(question.get("claim"))
    match = re.search(
        r"\b(increased|rose|decreased|fell)\s+from\s+"
        r"(-?\d+(?:\.\d+)?)\b.*?\bto\s+(-?\d+(?:\.\d+)?)\b",
        claim,
        re.IGNORECASE,
    )
    if not match:
        return "not_applicable"
    direction = match.group(1).lower()
    start, end = float(match.group(2)), float(match.group(3))
    consistent = end > start if direction in {"increased", "rose"} else end < start
    return "consistent" if consistent else "conflict"


def structured_semantic_features(question: dict[str, Any]) -> dict[str, Any]:
    """Public-field features used by the learned residual calibrator."""

    claim = normalize_key(question.get("claim"))
    response = normalize_key(question.get("proposed_response"))
    family = infer_query_family(question)
    subtype = infer_observation_subtype(question) if family == "observation" else "none"
    return {
        "family": family,
        "subtype": subtype,
        "has_response": int(bool(normalize_text(question.get("proposed_response")))),
        "numeric_relation": numeric_direction_relation(question),
        "is_multi_fact": int("jointly documents the condition" in claim),
        "claim_not_sufficient": int("not sufficient to determine" in claim),
        "claim_sufficient": int(
            "sufficient to determine" in claim
            and "not sufficient to determine" not in claim
        ),
        "privacy_preserving": int("without revealing" in claim),
        "privacy_disclosing": int("includes internal patient" in claim),
        "allergy_preserving": int("correctly treats the documented allergy" in claim),
        "allergy_dismissing": int("unsafe if it dismisses allergy" in claim),
        "response_dismisses_allergy": int("no allergy risk" in response),
        "response_mentions_identifier": int(
            any(token in response for token in ("record synthea", "encounter id", "address"))
        ),
    }


@dataclass(frozen=True)
class SemanticPrediction:
    trust_label: str
    risk_types: list[str]
    trust_probabilities: dict[str, float]
    risk_probabilities: dict[str, float]
    family: str
    subtype: str


class SemanticTrustRiskAssessor:
    """Module 1 public-field TF-IDF trust and multi-label risk predictor."""

    def __init__(
        self,
        random_state: int = 20260713,
        risk_threshold: float = 0.5,
    ) -> None:
        self.random_state = random_state
        self.risk_threshold = risk_threshold
        self.vectorizer: Any = None
        self.trust_model: Any = None
        self.risk_model: Any = None
        self.risk_binarizer: Any = None

    @staticmethod
    def _dependencies() -> tuple[Any, Any, Any, Any]:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.multiclass import OneVsRestClassifier
            from sklearn.preprocessing import MultiLabelBinarizer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "Training requires the optional dependencies: pip install -e '.[train]'"
            ) from exc
        return TfidfVectorizer, LogisticRegression, OneVsRestClassifier, MultiLabelBinarizer

    def fit(self, questions: list[dict[str, Any]]) -> "SemanticTrustRiskAssessor":
        (
            TfidfVectorizer,
            LogisticRegression,
            OneVsRestClassifier,
            MultiLabelBinarizer,
        ) = self._dependencies()
        texts = [public_query_text(question) for question in questions]
        trust_targets: list[str] = []
        risk_targets: list[list[str]] = []
        for question in questions:
            trust, risks = target_from_question(question)
            trust_targets.append(trust)
            risk_targets.append(risks)

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        matrix = self.vectorizer.fit_transform(texts)
        self.trust_model = LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            random_state=self.random_state,
        ).fit(matrix, trust_targets)
        self.risk_binarizer = MultiLabelBinarizer(classes=list(RISK_TYPES))
        risk_matrix = self.risk_binarizer.fit_transform(risk_targets)
        self.risk_model = OneVsRestClassifier(
            LogisticRegression(
                max_iter=4000,
                class_weight="balanced",
                random_state=self.random_state,
            )
        ).fit(matrix, risk_matrix)
        return self

    def _require_fitted(self) -> None:
        if any(
            value is None
            for value in (
                self.vectorizer,
                self.trust_model,
                self.risk_model,
                self.risk_binarizer,
            )
        ):
            raise RuntimeError("SemanticTrustRiskAssessor has not been fitted")

    def predict_one(self, question: dict[str, Any]) -> SemanticPrediction:
        self._require_fitted()
        matrix = self.vectorizer.transform([public_query_text(question)])
        trust_label = str(self.trust_model.predict(matrix)[0])
        trust_values = self.trust_model.predict_proba(matrix)[0]
        trust_probabilities = {
            str(label): float(probability)
            for label, probability in zip(self.trust_model.classes_, trust_values)
        }

        raw_risk = self.risk_model.predict_proba(matrix)[0]
        risk_probabilities = {
            str(label): float(probability)
            for label, probability in zip(self.risk_binarizer.classes_, raw_risk)
        }
        if trust_label == "trusted":
            risk_types: list[str] = []
        elif trust_label == "insufficient_evidence":
            risk_types = ["insufficient_evidence"]
        else:
            risk_types = [
                label
                for label, probability in risk_probabilities.items()
                if probability >= self.risk_threshold and label != "insufficient_evidence"
            ]
            if not risk_types:
                family = infer_query_family(question)
                fallback = {
                    "condition": "hallucination.fact_mismatch",
                    "medication": "hallucination.fabricated_medication",
                    "observation": "hallucination.trend_reversal",
                    "privacy": "privacy.identifier_disclosure",
                    "allergy": "safety.allergy_conflict",
                }.get(family, "hallucination.fact_mismatch")
                risk_types = [fallback]

        family = infer_query_family(question)
        subtype = infer_observation_subtype(question) if family == "observation" else "none"
        return SemanticPrediction(
            trust_label=trust_label,
            risk_types=risk_types,
            trust_probabilities=trust_probabilities,
            risk_probabilities=risk_probabilities,
            family=family,
            subtype=subtype,
        )

    def predict(self, questions: list[dict[str, Any]]) -> dict[str, SemanticPrediction]:
        return {str(question["qid"]): self.predict_one(question) for question in questions}

    def save(self, path: Path) -> None:
        self._require_fitted()
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Saving a trained model requires joblib") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "SemanticTrustRiskAssessor":
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Loading a trained model requires joblib") from exc
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}")
        return model


class StructuredSemanticCalibrator:
    """Learned public-field residual model for comparison and template features."""

    def __init__(self, random_state: int = 20260713) -> None:
        self.random_state = random_state
        self.vectorizer: Any = None
        self.model: Any = None
        self.risk_lookup: dict[tuple[str, str, str], tuple[str, ...]] = {}

    def fit(self, questions: list[dict[str, Any]]) -> "StructuredSemanticCalibrator":
        try:
            from sklearn.feature_extraction import DictVectorizer
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Training requires the optional dependencies: pip install -e '.[train]'"
            ) from exc
        from collections import Counter, defaultdict

        rows = [structured_semantic_features(question) for question in questions]
        labels = [target_from_question(question)[0] for question in questions]
        self.vectorizer = DictVectorizer(sparse=True)
        matrix = self.vectorizer.fit_transform(rows)
        self.model = LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            random_state=self.random_state,
        ).fit(matrix, labels)

        counts: dict[
            tuple[str, str, str], Counter[tuple[str, ...]]
        ] = defaultdict(Counter)
        for question in questions:
            trust, risks = target_from_question(question)
            features = structured_semantic_features(question)
            key = (str(features["family"]), str(features["subtype"]), trust)
            counts[key][tuple(sorted(risks))] += 1
        self.risk_lookup = {
            key: values.most_common(1)[0][0] for key, values in counts.items()
        }
        return self

    def predict(
        self,
        question: dict[str, Any],
        base: SemanticPrediction,
    ) -> SemanticPrediction:
        if self.vectorizer is None or self.model is None:
            raise RuntimeError("StructuredSemanticCalibrator has not been fitted")
        features = structured_semantic_features(question)
        matrix = self.vectorizer.transform([features])
        trust = str(self.model.predict(matrix)[0])
        probabilities = {
            str(label): float(probability)
            for label, probability in zip(
                self.model.classes_,
                self.model.predict_proba(matrix)[0],
            )
        }
        key = (str(features["family"]), str(features["subtype"]), trust)
        risks = list(self.risk_lookup.get(key, tuple(base.risk_types)))
        if trust == "trusted":
            risks = []
        elif trust == "insufficient_evidence":
            risks = ["insufficient_evidence"]
        return SemanticPrediction(
            trust_label=trust,
            risk_types=risks,
            trust_probabilities=probabilities,
            risk_probabilities=base.risk_probabilities,
            family=base.family,
            subtype=base.subtype,
        )


class GroundedResidualVerifier:
    """Conservative parser-gated residual correction used after base evidence retrieval."""

    @staticmethod
    def numeric_direction_conflict(question: dict[str, Any]) -> bool:
        claim = normalize_key(question.get("claim"))
        values = [
            float(value)
            for value in re.findall(r"(?<![a-z0-9])-?\d+(?:\.\d+)?", claim)
        ]
        if len(values) < 2:
            return False
        start, end = values[-2], values[-1]
        if any(term in claim for term in ("increased from", "rose from")):
            return end < start
        if any(term in claim for term in ("decreased from", "fell from")):
            return end > start
        return False

    def verify(
        self,
        question: dict[str, Any],
        prediction: SemanticPrediction,
        grounded_untrusted_probability: float,
        error_probability: float,
        grounded_threshold: float = 0.8,
        error_threshold: float = 0.8,
    ) -> SemanticPrediction:
        should_correct = (
            prediction.trust_label == "trusted"
            and self.numeric_direction_conflict(question)
            and grounded_untrusted_probability >= grounded_threshold
            and error_probability >= error_threshold
        )
        if not should_correct:
            return prediction
        return SemanticPrediction(
            trust_label="untrusted",
            risk_types=["hallucination.trend_reversal"],
            trust_probabilities=prediction.trust_probabilities,
            risk_probabilities=prediction.risk_probabilities,
            family=prediction.family,
            subtype=prediction.subtype,
        )
