# TriageDesk

TriageDesk is a small, production-engineered support-message intent classifier. It turns a short English banking support message into a suggested intent, queue, probability estimate, and a human-review decision. The Streamlit workbench is a client of a FastAPI inference service; it never loads the model itself.

The repository includes the evaluated 77-intent release artifact, typed FastAPI contracts, a Streamlit client, bounded batch processing, deployment configuration, and tests. The implementation is designed as a low-traffic portfolio service: uncertain or policy-sensitive requests remain explicitly reviewable.

## Live deployment

- **Try the Streamlit workbench:** https://triagedesk-ui.onrender.com
- **FastAPI readiness:** https://triagedesk-api-fn3x.onrender.com/health/ready
- **Repository:** https://github.com/Hitman45-coder/triagedesk-support-routing

The public release runs on Render's free profile. It can sleep after inactivity, so the first request after a quiet period may take longer. The model bundle is included in the API image; user messages are held only in the Streamlit session and are not stored by the API.

## Quick start

```bash
uv sync --extra dev
uv run python -m triagedesk.cli train-demo
cp .env.example .env
uv run uvicorn triagedesk.serving.app:app --reload --port 8000
```

In another terminal:

```bash
uv run streamlit run ui/app.py
```

Open `http://localhost:8501`. The local API key is read by Streamlit from `INFERENCE_API_KEY`; it is never sent to browser JavaScript.

## Current scope

The repository includes the evaluated release artifact, so a clean checkout can start the full model without downloading training data. To rebuild it from the pinned public source, run:

```bash
uv run python -m triagedesk.cli fetch-data --raw-dir data/raw/banking77 --commit 57ec275d8078af65b7731c2a98be812d844a6d6b
uv run python -m triagedesk.cli train --raw-dir data/raw/banking77 --output artifacts/releases/banking77
MODEL_BUNDLE_PATH=artifacts/releases/banking77 uv run uvicorn triagedesk.serving.app:app --port 8000
```

The current full artifact has 77 labels and an observed 0.8942 macro-F1 on the official 3,080-row test file. It uses a separate calibration split and a policy-validation split; the frozen test policy accepted 77.2% of messages with 96.0% selective accuracy. These are artifact-level benchmark measurements, not a final production guarantee.

The product is a routing recommendation tool. It does not contact a bank, move money, resolve a complaint, or provide banking advice. User messages are held only in the Streamlit session and are not stored by the API.

- Product and implementation specification: [plan.md](plan.md)
- Architecture: [docs/architecture.md](docs/architecture.md)
- Evaluation report: [docs/evaluation.md](docs/evaluation.md)
- Current implementation evidence: [docs/implementation-status.md](docs/implementation-status.md)
- Data attribution and license: [DATA_LICENSE.md](DATA_LICENSE.md)
- Security boundary: [SECURITY.md](SECURITY.md)
