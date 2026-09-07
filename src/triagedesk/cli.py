from __future__ import annotations

import argparse
import json
from pathlib import Path

from triagedesk.data.pipeline import fetch_banking77, read_rows, rows_to_xy, validate_banking77
from triagedesk.ml.artifacts import load_routes, write_manifest
from triagedesk.ml.model import build_pipeline, train_demo

ROOT = Path(__file__).resolve().parents[2]
ROUTES_PATH = ROOT / "configs" / "routes.json"


def train_demo_command(output: Path) -> None:
    routes = load_routes(ROUTES_PATH)
    bundle = train_demo(routes)
    bundle.save(output)
    write_manifest(output, bundle)
    print(f"wrote demo artifact to {output}")


def fetch_data_command(raw_dir: Path, commit: str) -> None:
    manifest = fetch_banking77(raw_dir, commit=commit)
    print(f"fetched Banking77 snapshot: {manifest['files']}")


def train_banking77_command(raw_dir: Path, output: Path) -> None:
    import numpy as np
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    summary = validate_banking77(raw_dir)
    source_manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    train_rows = read_rows(raw_dir / "train.csv")
    test_rows = read_rows(raw_dir / "test.csv")
    test_text_keys = {" ".join(row["text"].casefold().split()) for row in test_rows}
    train_rows = [
        row for row in train_rows if " ".join(row["text"].casefold().split()) not in test_text_keys
    ]
    fit_rows, heldout_rows = train_test_split(
        train_rows,
        test_size=0.30,
        random_state=42,
        stratify=[row["category"] for row in train_rows],
    )
    calibration_rows, policy_rows = train_test_split(
        heldout_rows,
        test_size=0.50,
        random_state=42,
        stratify=[row["category"] for row in heldout_rows],
    )
    fit_texts, fit_labels = rows_to_xy(fit_rows)
    calibration_texts, calibration_labels = rows_to_xy(calibration_rows)
    policy_texts, policy_labels = rows_to_xy(policy_rows)
    test_texts, test_labels = rows_to_xy(test_rows)
    pipeline = build_pipeline().fit(fit_texts, fit_labels)
    calibrated = CalibratedClassifierCV(FrozenEstimator(pipeline), method="sigmoid")
    calibrated.fit(calibration_texts, calibration_labels)
    routes = load_routes(ROUTES_PATH)
    categories = json.loads((raw_dir / "categories.json").read_text(encoding="utf-8"))
    routes = build_routes(categories, routes)

    def score(texts: list[str], labels: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        probabilities = calibrated.predict_proba(texts)
        order = np.argsort(probabilities, axis=1)[:, ::-1]
        top_indices = order[:, 0]
        predictions = np.asarray(calibrated.classes_)[top_indices]
        confidence = probabilities[np.arange(len(probabilities)), top_indices]
        second = probabilities[np.arange(len(probabilities)), order[:, 1]]
        return predictions, confidence, confidence - second

    policy_predictions, policy_confidence, policy_margin = score(policy_texts, policy_labels)
    candidates: list[tuple[float, float, float, float]] = []
    for confidence_threshold in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        for margin_threshold in (0.05, 0.10, 0.15, 0.20, 0.25):
            accepted = np.array([
                not routes[label]["mandatory_review"]
                and confidence >= confidence_threshold
                and margin >= margin_threshold
                for label, confidence, margin in zip(policy_predictions, policy_confidence, policy_margin, strict=True)
            ])
            count = int(accepted.sum())
            selective_accuracy = float((policy_predictions[accepted] == np.asarray(policy_labels)[accepted]).mean()) if count else 0.0
            coverage = float(accepted.mean())
            if count >= 200 and selective_accuracy >= 0.95:
                candidates.append((coverage, selective_accuracy, confidence_threshold, margin_threshold))
    if candidates:
        _, _, confidence_threshold, margin_threshold = max(candidates)
        automation_enabled = True
    else:
        confidence_threshold, margin_threshold, automation_enabled = 0.75, 0.20, False

    predicted, confidence, margin = score(test_texts, test_labels)
    test_accepted = np.array([
        automation_enabled
        and not routes[label]["mandatory_review"]
        and score_confidence >= confidence_threshold
        and score_margin >= margin_threshold
        for label, score_confidence, score_margin in zip(predicted, confidence, margin, strict=True)
    ])
    accepted_count = int(test_accepted.sum())
    test_selective_accuracy = float((predicted[test_accepted] == np.asarray(test_labels)[test_accepted]).mean()) if accepted_count else None
    from triagedesk.ml.model import ModelBundle

    bundle = ModelBundle(
        model=calibrated,
        classes=list(calibrated.classes_),
        model_version="banking77-0.1.0",
        policy_version="policy-0.1.0",
        routes=routes,
        examples=[
            {"example_id": f"train-{i:05d}", "text": row["text"], "intent": row["category"]}
            for i, row in enumerate(fit_rows[:2000])
        ],
        metrics={
            "scope": "banking77-test",
            "dataset_commit": str(source_manifest.get("commit", "unknown")),
            "accuracy": float(accuracy_score(test_labels, predicted)),
            "macro_f1": float(f1_score(test_labels, predicted, average="macro")),
            "train_rows": summary["train_rows"],
            "train_rows_after_test_duplicate_removal": len(train_rows),
            "fit_rows": len(fit_rows),
            "calibration_rows": len(calibration_rows),
            "policy_rows": len(policy_rows),
            "test_rows": summary["test_rows"],
            "normalized_train_test_duplicates": summary["normalized_train_test_duplicates"],
            "policy_validation_candidates": len(candidates),
            "test_automatic_coverage": float(test_accepted.mean()),
            "test_selective_accuracy": test_selective_accuracy,
            "test_automatic_count": accepted_count,
            "label_count": summary["categories"],
        },
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
        automation_enabled=automation_enabled,
    )
    bundle.save(output)
    write_manifest(output, bundle)
    print(f"wrote Banking77 artifact to {output}")


def build_routes(categories: list[str], configured: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    security_markers = ("compromised", "lost_or_stolen", "not_recognised", "not_recognized")

    def queue_for(label: str) -> str:
        if "cash_withdrawal" in label or "cash" in label or "atm" in label:
            return "Cash & ATMs"
        if "transfer" in label or "beneficiary" in label or "receiving_money" in label:
            return "Transfers"
        if "top_up" in label or "topping_up" in label:
            return "Top-ups"
        if "identity" in label or "verify" in label or "source_of_funds" in label:
            return "Account & Verification"
        if "payment" in label or "refund" in label or "exchange" in label:
            return "Card Payments"
        if "card" in label or "pin" in label or "contactless" in label or "virtual" in label:
            return "Cards"
        return "Account & Verification"

    return {
        label: configured.get(
            label,
            {
                "display_name": label.replace("_", " ").title(),
                "queue": queue_for(label),
                "mandatory_review": any(marker in label for marker in security_markers),
            },
        )
        for label in categories
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="triagedesk")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("train-demo", help="train the tiny local bootstrap artifact")
    demo.add_argument("--output", type=Path, default=ROOT / "artifacts" / "releases" / "demo")
    fetch = subparsers.add_parser(
        "fetch-data", help="fetch and validate the pinned Banking77 snapshot"
    )
    fetch.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw" / "banking77")
    fetch.add_argument("--commit", default="57ec275d8078af65b7731c2a98be812d844a6d6b")
    train = subparsers.add_parser("train", help="train a full Banking77 artifact")
    train.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw" / "banking77")
    train.add_argument("--output", type=Path, default=ROOT / "artifacts" / "releases" / "banking77")
    args = parser.parse_args()
    if args.command == "train-demo":
        train_demo_command(args.output)
    elif args.command == "fetch-data":
        fetch_data_command(args.raw_dir, args.commit)
    elif args.command == "train":
        train_banking77_command(args.raw_dir, args.output)


if __name__ == "__main__":
    main()
