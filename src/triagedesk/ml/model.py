from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from triagedesk.ml.preprocess import normalize_text

DEMO_ROWS: list[tuple[str, str]] = [
    ("My replacement card still has not arrived", "card_arrival"),
    ("How do I activate my new card", "activate_my_card"),
    ("My card stopped working at the shop", "card_not_working"),
    ("I do not recognize this card payment", "card_payment_not_recognised"),
    ("Someone stole my card", "lost_or_stolen_card"),
    ("My card may be compromised", "compromised_card"),
    ("The transfer is still pending", "pending_transfer"),
    ("Why did my transfer fail", "failed_transfer"),
    ("The recipient has not received my transfer", "transfer_not_received_by_recipient"),
    ("How long does a transfer take", "transfer_timing"),
    ("My cash withdrawal was declined", "declined_cash_withdrawal"),
    ("The ATM charged me a fee", "cash_withdrawal_charge"),
    ("I received the wrong amount of cash", "wrong_amount_of_cash_received"),
    ("My top up failed", "top_up_failed"),
    ("My top up is pending", "pending_top_up"),
    ("How can I top up by card", "topping_up_by_card"),
    ("My card payment was declined", "declined_card_payment"),
    ("I need to request a refund", "request_refund"),
    ("I need to verify my identity", "verify_my_identity"),
    ("I cannot verify my identity", "unable_to_verify_identity"),
    ("I forgot my passcode", "passcode_forgotten"),
    ("I need to change my personal details", "edit_personal_details"),
]


@dataclass
class ModelBundle:
    model: Any
    classes: list[str]
    model_version: str
    policy_version: str
    routes: dict[str, dict[str, Any]]
    examples: list[dict[str, str]]
    metrics: dict[str, float | int | str | None]
    confidence_threshold: float = 0.55
    margin_threshold: float = 0.10
    automation_enabled: bool = True
    artifact_schema_version: str = "1.0"

    def predict(self, text: str, top_k: int = 3) -> list[tuple[str, float]]:
        clean = normalize_text(text)
        probabilities = self.model.predict_proba([clean])[0]
        order = np.argsort(probabilities)[::-1][:top_k]
        return [(self.classes[int(i)], float(probabilities[int(i)])) for i in order]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path / "bundle.joblib", compress=3)

    @classmethod
    def load(cls, path: Path) -> ModelBundle:
        bundle = joblib.load(path / "bundle.joblib")
        if not isinstance(bundle, cls):
            raise ValueError("artifact is not a TriageDesk ModelBundle")
        if not bundle.classes or not bundle.routes:
            raise ValueError("artifact is missing classes or routes")
        return bundle


def build_pipeline() -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_features=30000
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=50000
                ),
            ),
        ]
    )
    base = LogisticRegression(max_iter=1200, C=4.0, solver="lbfgs")
    return Pipeline([("features", features), ("classifier", base)])


def train_demo(routes: dict[str, dict[str, Any]]) -> ModelBundle:
    base_texts = [normalize_text(row[0]) for row in DEMO_ROWS]
    base_labels = [row[1] for row in DEMO_ROWS]
    texts = base_texts * 3
    labels = base_labels * 3
    pipeline = build_pipeline().fit(texts, labels)
    classes = list(pipeline.named_steps["classifier"].classes_)
    filtered_routes = {
        label: routes.get(
            label, {"display_name": label, "queue": "Human Review", "mandatory_review": True}
        )
        for label in classes
    }
    examples = [
        {"example_id": f"demo-{i:03d}", "text": text, "intent": label}
        for i, (text, label) in enumerate(DEMO_ROWS)
    ]
    return ModelBundle(
        model=pipeline,
        classes=classes,
        model_version="demo-0.1.0",
        policy_version="policy-demo-0.1.0",
        routes=filtered_routes,
        examples=examples,
        metrics={
            "scope": "demo-only",
            "accuracy": None,
            "macro_f1": None,
            "label_count": len(classes),
        },
    )
