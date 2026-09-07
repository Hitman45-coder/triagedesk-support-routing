import json
from pathlib import Path
from typing import Any


def load_routes(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("routes file must contain an object")
    return payload


def write_manifest(path: Path, bundle: Any) -> None:
    manifest = {
        "artifact_schema_version": bundle.artifact_schema_version,
        "model_version": bundle.model_version,
        "policy_version": bundle.policy_version,
        "label_count": len(bundle.classes),
        "metrics": bundle.metrics,
        "scope": "demo-only" if bundle.model_version.startswith("demo-") else "banking77",
        "confidence_threshold": bundle.confidence_threshold,
        "margin_threshold": bundle.margin_threshold,
        "automation_enabled": bundle.automation_enabled,
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
