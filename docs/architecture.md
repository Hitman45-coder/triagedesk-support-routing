# TriageDesk architecture

The browser talks to Streamlit. Streamlit sends a server-side authenticated HTTP request to FastAPI. FastAPI loads one verified model bundle at startup, validates the request, applies the persisted routing policy, and returns a typed response. The API does not store messages.

Offline training fetches the pinned source snapshot, validates the two CSV files, audits normalized duplicates, creates fitting/calibration/policy partitions, trains the sparse classifier, calibrates probabilities, selects a policy, evaluates the official test set, and writes a bundle plus manifest. Runtime startup loads that bundle; it never downloads data or retrains.

The deployment images are intentionally split. The UI image contains no model and the API image owns inference. Compose runs them on a private local network. `render.yaml` defines the same two-service topology for a free public portfolio deployment: Render builds the images from the repository, probes API readiness through `/health/ready`, and injects the API host and generated inference key into the UI. The first public release is live at `https://triagedesk-ui.onrender.com` with the API at `https://triagedesk-api-fn3x.onrender.com`.

The free profile is suitable for a low-traffic portfolio demonstration. Services can sleep after inactivity and their filesystems are ephemeral; the release model is therefore packaged into the API image, while messages remain session-local in Streamlit and are not persisted.
