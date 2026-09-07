# Security boundary

TriageDesk is a low-traffic portfolio demo. The API uses a server-to-server bearer key, strict request schemas, bounded batch sizes, body limits, rate limiting, and redacted structured logs. Streamlit keeps the key on the server side.

Do not submit real account numbers, card numbers, passwords, or other sensitive banking information. Messages are processed for the demo and are not a durable customer database. A production banking deployment would require authenticated users, access control, retention/deletion, audit storage, privacy review, and representative data.

Report suspected vulnerabilities privately to the repository owner. Never commit secrets or serialized artifacts from untrusted sources.
