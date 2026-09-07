from __future__ import annotations

import argparse
import json
import os
from urllib.request import Request, urlopen


def request(base_url: str, path: str, key: str, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = Request(f"{base_url.rstrip('/')}{path}", data=body, headers=headers)
    with urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("API_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--api-key", default=os.getenv("INFERENCE_API_KEY"))
    args = parser.parse_args()
    if not args.api_key:
        raise SystemExit("set INFERENCE_API_KEY or pass --api-key")
    assert request(args.base_url, "/health/live", "")["status"] == "ok"
    assert request(args.base_url, "/health/ready", "")["status"] == "ready"
    result = request(args.base_url, "/api/v1/triage", args.api_key, {"text": "My replacement card still has not arrived"})
    assert result["predicted_intent"]
    assert result["model_version"]
    print(json.dumps({"status": "ok", "model_version": result["model_version"], "disposition": result["disposition"]}))


if __name__ == "__main__":
    main()
