from __future__ import annotations

import math
import time
from collections import deque
from threading import Lock, Semaphore

from triagedesk.contracts import Alternative, SimilarExample, TriageResponse
from triagedesk.ml.model import ModelBundle
from triagedesk.ml.preprocess import normalize_text


class RateLimiter:
    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._events: deque[float] = deque()
        self._lock = Lock()

    def allow(self, cost: int = 1) -> bool:
        now = time.monotonic()
        with self._lock:
            while self._events and now - self._events[0] >= self.window_seconds:
                self._events.popleft()
            if len(self._events) + cost > self.limit:
                return False
            self._events.extend([now] * cost)
            return True


class TriageService:
    def __init__(self, bundle: ModelBundle, max_in_flight: int = 2, rate_limit: int = 300) -> None:
        self.bundle = bundle
        self.slots = Semaphore(max_in_flight)
        self.rate_limiter = RateLimiter(rate_limit)

    def _similar_examples(self, text: str, limit: int = 3) -> list[SimilarExample]:
        # Bootstrap retrieval is deterministic token overlap. Production replaces this
        # with cosine similarity over a frozen sparse reference matrix.
        query = set(normalize_text(text).lower().split())
        ranked: list[tuple[float, dict[str, str]]] = []
        for example in self.bundle.examples:
            tokens = set(example["text"].lower().split())
            union = query | tokens
            score = len(query & tokens) / len(union) if union else 0.0
            ranked.append((score, example))
        ranked.sort(key=lambda item: (-item[0], item[1]["example_id"]))
        return [
            SimilarExample(
                example_id=example["example_id"],
                text=example["text"],
                intent=example["intent"],
                similarity=round(score, 4),
            )
            for score, example in ranked[:limit]
            if score > 0
        ]

    def triage(self, text: str, request_id: str, include_examples: bool = False) -> TriageResponse:
        started = time.perf_counter()
        normalized = normalize_text(text)
        if len(normalized) < 3:
            return TriageResponse(
                request_id=request_id,
                model_version=self.bundle.model_version,
                policy_version=self.bundle.policy_version,
                predicted_intent=None,
                suggested_queue=None,
                recommended_queue="Human Review",
                disposition="review_required",
                reason_codes=["no_supported_features"],
                confidence=None,
                margin=None,
                alternatives=[],
                processing_ms=(time.perf_counter() - started) * 1000,
            )
        if not self.slots.acquire(blocking=False):
            raise RuntimeError("inference_busy")
        try:
            predictions = self.bundle.predict(normalized, top_k=3)
        finally:
            self.slots.release()
        top_intent, confidence = predictions[0]
        second = predictions[1][1] if len(predictions) > 1 else 0.0
        margin = confidence - second
        route = self.bundle.routes.get(
            top_intent,
            {"display_name": top_intent, "queue": "Human Review", "mandatory_review": True},
        )
        reasons: list[str] = []
        if bool(route.get("mandatory_review")):
            reasons.append("policy_review")
        if confidence < self.bundle.confidence_threshold:
            reasons.append("low_confidence")
        if margin < self.bundle.margin_threshold:
            reasons.append("ambiguous_intent")
        if not self.bundle.automation_enabled:
            reasons.append("assist_only")
        disposition = "review_required" if reasons else "auto_route_recommended"
        recommended_queue = (
            route.get("queue", "Human Review")
            if disposition == "auto_route_recommended"
            else "Human Review"
        )
        alternatives = [
            Alternative(
                intent=intent,
                display_name=str(self.bundle.routes.get(intent, {}).get("display_name", intent)),
                probability=round(probability, 6),
            )
            for intent, probability in predictions
        ]
        return TriageResponse(
            request_id=request_id,
            model_version=self.bundle.model_version,
            policy_version=self.bundle.policy_version,
            predicted_intent=top_intent,
            suggested_queue=route.get("queue"),
            recommended_queue=recommended_queue,
            disposition=disposition,
            reason_codes=reasons,
            confidence=round(confidence, 6) if math.isfinite(confidence) else None,
            margin=round(margin, 6) if math.isfinite(margin) else None,
            alternatives=alternatives,
            similar_examples=self._similar_examples(normalized) if include_examples else [],
            processing_ms=round((time.perf_counter() - started) * 1000, 3),
        )
