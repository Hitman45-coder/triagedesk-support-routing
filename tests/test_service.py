from pathlib import Path

from triagedesk.ml.artifacts import load_routes
from triagedesk.ml.model import train_demo
from triagedesk.serving.service import TriageService


def make_service() -> TriageService:
    routes = load_routes(Path("configs/routes.json"))
    return TriageService(train_demo(routes))


def test_security_intent_requires_review():
    result = make_service().triage("I do not recognize this card payment", "test-1")
    assert result.disposition == "review_required"
    assert "policy_review" in result.reason_codes
    assert result.recommended_queue == "Human Review"


def test_short_text_requires_review_without_prediction():
    result = make_service().triage("?", "test-2")
    assert result.predicted_intent is None
    assert result.reason_codes == ["no_supported_features"]
