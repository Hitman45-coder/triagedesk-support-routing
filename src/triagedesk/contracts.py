from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class TriageRequest(StrictModel):
    text: str = Field(min_length=1, max_length=2000)
    include_examples: bool = False


class BatchItem(StrictModel):
    text: str = Field(min_length=1, max_length=2000)
    client_id: str | None = Field(default=None, max_length=100)


class BatchRequest(StrictModel):
    items: list[BatchItem] = Field(min_length=1, max_length=100)


class Alternative(StrictModel):
    intent: str
    display_name: str
    probability: float = Field(ge=0, le=1)


class SimilarExample(StrictModel):
    example_id: str
    text: str
    intent: str
    similarity: float = Field(ge=0, le=1)


Disposition = Literal["auto_route_recommended", "review_required"]


class TriageResponse(StrictModel):
    request_id: str
    model_version: str
    policy_version: str
    predicted_intent: str | None
    suggested_queue: str | None
    recommended_queue: str
    disposition: Disposition
    reason_codes: list[str]
    confidence: float | None = Field(default=None, ge=0, le=1)
    margin: float | None = Field(default=None, ge=-1, le=1)
    alternatives: list[Alternative]
    similar_examples: list[SimilarExample] = Field(default_factory=list)
    processing_ms: float = Field(ge=0)


class BatchResult(StrictModel):
    row_index: int = Field(ge=0)
    client_id: str | None = None
    ok: bool
    result: TriageResponse | None = None
    error_code: str | None = None
    error_message: str | None = None


class BatchResponse(StrictModel):
    request_id: str
    model_version: str
    policy_version: str
    results: list[BatchResult]
    total: int = Field(ge=0)
    valid: int = Field(ge=0)
    invalid: int = Field(ge=0)
    processing_ms: float = Field(ge=0)


class ModelMetadata(StrictModel):
    model_version: str
    policy_version: str
    label_count: int = Field(ge=1)
    artifact_schema_version: str
    scope: str
    metrics: dict[str, float | int | str | None]


class IntentMetadata(StrictModel):
    intent: str
    display_name: str
    queue: str
    mandatory_review: bool
