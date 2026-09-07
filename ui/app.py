from __future__ import annotations

import os
from io import StringIO

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
if not API_BASE_URL.startswith(("http://", "https://")):
    API_BASE_URL = f"https://{API_BASE_URL}"
API_KEY = os.getenv("INFERENCE_API_KEY", "local-dev-key-change-me")
TIMEOUT = httpx.Timeout(connect=3.0, read=15.0, write=5.0, pool=3.0)

st.set_page_config(page_title="TriageDesk", page_icon="🎫", layout="wide")
st.title("TriageDesk")
st.caption("Support-message routing suggestions with explicit human review for uncertain cases.")
st.info(
    "English banking messages only. This portfolio demo suggests a queue; it does not contact a bank or move money."
)


def api_post(path: str, payload: dict) -> dict:
    try:
        response = httpx.post(
            f"{API_BASE_URL}{path}",
            json=payload,
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.json().get("error", {}).get("message", "The API rejected the request")
        raise RuntimeError(f"{detail} (HTTP {exc.response.status_code})") from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(
            "The inference service is unavailable. Start the API and retry."
        ) from exc


def safe_csv_cell(value: object) -> str:
    text_value = "" if value is None else str(value)
    if text_value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + text_value
    return text_value


def export_rows(rows: list[dict[str, object]], include_text: bool) -> bytes:
    import csv

    output = StringIO(newline="")
    fields = ["row_index", "client_id", "intent", "queue", "disposition", "confidence", "reviewer_intent"]
    if include_text:
        fields.insert(2, "text")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: safe_csv_cell(row.get(field, "")) for field in fields})
    return output.getvalue().encode("utf-8")


if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "batch_rows" not in st.session_state:
    st.session_state.batch_rows = []

sample = st.selectbox(
    "Try a sample",
    [
        "Choose a sample",
        "My replacement card still has not arrived",
        "I do not recognize this card payment",
        "The transfer is still pending",
    ],
)
default_text = "" if sample == "Choose a sample" else sample
text = st.text_area(
    "Support message",
    value=default_text,
    max_chars=2000,
    height=140,
    placeholder="Example: My replacement card has not arrived yet",
)

if st.button("Analyze message", type="primary", disabled=not text.strip()):
    with st.spinner("Analyzing with the model..."):
        try:
            st.session_state.last_result = api_post(
                "/api/v1/triage", {"text": text, "include_examples": True}
            )
        except RuntimeError as exc:
            st.error(str(exc))

result = st.session_state.last_result
if result:
    st.divider()
    if result["disposition"] == "review_required":
        st.warning("Human review recommended")
    else:
        st.success("Automatic routing recommendation")
    left, middle, right = st.columns(3)
    left.metric("Predicted intent", result.get("predicted_intent") or "Needs review")
    middle.metric("Recommended queue", result["recommended_queue"])
    confidence = result.get("confidence")
    right.metric("Confidence", f"{confidence:.1%}" if confidence is not None else "N/A")
    if result.get("reason_codes"):
        st.caption("Reason codes: " + ", ".join(result["reason_codes"]))
    st.subheader("Alternatives")
    st.dataframe(result.get("alternatives", []), use_container_width=True, hide_index=True)
    with st.expander("Similar training examples"):
        examples = result.get("similar_examples", [])
        if examples:
            st.dataframe(examples, use_container_width=True, hide_index=True)
        else:
            st.caption("No eligible similar examples were found.")
    st.caption(
        f"Model {result['model_version']} · policy {result['policy_version']} · {result['processing_ms']} ms"
    )

st.divider()
st.subheader("Batch workbench")
st.caption(
    "Upload a CSV with a required `text` column. Files are processed for this session and are not stored by the API."
)
uploaded = st.file_uploader("CSV file", type=["csv"])
if uploaded:
    import pandas as pd

    try:
        frame = pd.read_csv(uploaded)
        if "text" not in frame.columns:
            st.error("CSV must include a `text` column.")
        elif len(frame) > 100:
            st.error("The demo accepts at most 100 rows.")
        else:
            st.dataframe(frame.head(10), use_container_width=True, hide_index=True)
            if st.button("Analyze batch"):
                payload = {
                    "items": [
                        {"text": str(value), "client_id": str(i)}
                        for i, value in enumerate(frame["text"].fillna(""))
                    ]
                }
                try:
                    batch = api_post("/api/v1/triage/batch", payload)
                    rows = []
                    for item in batch["results"]:
                        row = {
                            "row_index": item["row_index"],
                            "client_id": item.get("client_id"),
                            "ok": item["ok"],
                            "text": str(frame.iloc[item["row_index"]]["text"]),
                            "reviewer_intent": "",
                        }
                        if item["ok"]:
                            row.update(
                                {
                                    "intent": item["result"]["predicted_intent"],
                                    "queue": item["result"]["recommended_queue"],
                                    "disposition": item["result"]["disposition"],
                                    "confidence": item["result"]["confidence"],
                                    "alternatives": item["result"].get("alternatives", []),
                                }
                            )
                        else:
                            row["error"] = item.get("error_message")
                        rows.append(row)
                    st.session_state.batch_rows = rows
                except RuntimeError as exc:
                    st.error(str(exc))
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")

    if st.session_state.batch_rows:
        st.subheader("Review results")
        disposition_filter = st.multiselect(
            "Show dispositions",
            ["auto_route_recommended", "review_required", "error"],
            default=["auto_route_recommended", "review_required", "error"],
        )
        visible_rows = [
            row
            for row in st.session_state.batch_rows
            if ("error" if not row.get("ok") else row.get("disposition")) in disposition_filter
        ]
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"text", "ok", "reviewer_intent"}
                    }
                    for row in visible_rows
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        editable = [row for row in st.session_state.batch_rows if row.get("ok")]
        if editable:
            selected_index = st.selectbox(
                "Choose a row to correct",
                [int(row["row_index"]) for row in editable],
            )
            selected = next(row for row in editable if row["row_index"] == selected_index)
            correction_options = [selected.get("intent", "")] + [
                alternative["intent"] for alternative in selected.get("alternatives", [])
            ]
            correction = st.selectbox("Reviewer intent (session only)", sorted(set(correction_options)))
            if st.button("Save review decision"):
                selected["reviewer_intent"] = correction
                st.success("Saved in this browser session. It is not training data.")
        include_text = st.checkbox("Include message text in export", value=False)
        st.download_button(
            "Download reviewed CSV",
            data=export_rows(st.session_state.batch_rows, include_text),
            file_name="triagedesk-reviewed.csv",
            mime="text/csv",
        )
with st.expander("About this project"):
    st.write(
        "The full implementation plan, data attribution, evaluation, and limitations are available in the repository. This deployment serves the pinned 77-intent BANKING77 artifact and keeps reviewer corrections in the current browser session only."
    )
