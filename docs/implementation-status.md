# Implementation status

Updated: 2026-09-07

## Completed

- Repository metadata, dependency intent, environment template, package boundaries.
- Typed FastAPI contracts and a real model-loading lifespan.
- Deterministic bootstrap artifact training command (`train-demo`).
- Single-message and bounded batch endpoints with authentication, readiness, schema limits, rate limiting, and redacted request logs.
- Streamlit client that calls the API over HTTP, displays review decisions, alternatives, examples, and batch results.
- Streamlit batch session review with disposition filters, corrections, and formula-safe CSV export.
- Local Compose files and initial unit tests.
- Public BANKING77 snapshot fetched into ignored raw data with SHA-256 manifest.
- First full 77-intent artifact trained after removing seven normalized train/test duplicates from development rows.
- Full artifact now uses disjoint fit/calibration/policy partitions and persists calibrated probabilities plus routing thresholds.
- Real HTTP smoke verified readiness, model metadata, authenticated single-message inference, and batch inference against the 77-intent artifact.

## Not complete yet

- Artifact checksum verification and immutable release promotion.
- Full API contract, browser, security, load, and deployment tests.
- Public deployment and rollback evidence are recorded after the first Render release.

## Verification

Run after dependencies are installed:

```bash
uv sync --extra dev
uv run python -m triagedesk.cli train-demo
uv run pytest
```

For the current full artifact:

```bash
uv run python -m triagedesk.cli fetch-data --raw-dir data/raw/banking77 --commit 57ec275d8078af65b7731c2a98be812d844a6d6b
uv run python -m triagedesk.cli train --raw-dir data/raw/banking77 --output artifacts/releases/banking77
MODEL_BUNDLE_PATH=artifacts/releases/banking77 uv run uvicorn triagedesk.serving.app:app --port 8000
```

Observed on the official test file for the current calibrated artifact: accuracy `0.8951`, macro-F1 `0.8942`, 77 labels, and 3,080 test rows. The frozen policy accepted `77.2%` of messages with `96.0%` selective accuracy (`2,379` accepted). The seven normalized duplicates are recorded in the artifact manifest; they were removed from development training while the official test remained unchanged.

The demo artifact is intentionally not benchmark evidence. The current live HTTP smoke was run with the full artifact and verified `/api/v1/model`, single triage, and batch triage.
