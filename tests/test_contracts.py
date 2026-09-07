import pytest
from pydantic import ValidationError

from triagedesk.contracts import BatchRequest, TriageRequest


def test_request_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        TriageRequest(text="hello", extra_field="nope")


def test_batch_enforces_non_empty_bounded_items():
    request = BatchRequest(items=[{"text": "hello", "client_id": "row-1"}])
    assert request.items[0].client_id == "row-1"
    with pytest.raises(ValidationError):
        BatchRequest(items=[])
