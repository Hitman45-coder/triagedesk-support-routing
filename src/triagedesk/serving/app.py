from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from triagedesk.contracts import (
    BatchRequest,
    BatchResponse,
    BatchResult,
    IntentMetadata,
    ModelMetadata,
    TriageRequest,
    TriageResponse,
)
from triagedesk.ml.model import ModelBundle
from triagedesk.serving.service import TriageService
from triagedesk.settings import Settings, get_settings

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("triagedesk.api")


def _load_service(settings: Settings) -> TriageService:
    path = Path(settings.model_bundle_path)
    bundle = ModelBundle.load(path)
    return TriageService(bundle, settings.max_in_flight, settings.rate_limit_per_minute)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.service = _load_service(settings)
    app.state.ready = True
    yield
    app.state.ready = False


app = FastAPI(title="TriageDesk API", version="0.1.0", lifespan=lifespan)


def request_id(request: Request) -> str:
    return request.headers.get("x-request-id", str(uuid.uuid4()))


def log_inference(current_id: str, batch_size: int, status_value: str, disposition_count: int = 0) -> None:
    logger.info(
        '{"event":"inference","request_id":"%s","batch_size":%s,"status":"%s","disposition_count":%s}',
        current_id,
        batch_size,
        status_value,
        disposition_count,
    )


def current_service(request: Request) -> TriageService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="service_not_ready")
    return service


def require_key(request: Request, authorization: str | None = Header(default=None)) -> None:
    settings: Settings = request.app.state.settings
    if authorization != f"Bearer {settings.inference_api_key}":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_credentials")


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    current_id = request_id(request)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": str(exc.detail),
                "message": "Request could not be completed",
                "request_id": current_id,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    del exc
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": "Request fields are invalid or exceed the allowed limits",
                "request_id": request_id(request),
            }
        },
    )


@app.get("/health/live")
def live():
    return {"status": "ok"}


@app.get("/health/ready")
def ready(request: Request):
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="service_not_ready")
    return {"status": "ready"}


@app.get("/api/v1/model", response_model=ModelMetadata)
def model_metadata(service: TriageService = Depends(current_service)):
    bundle = service.bundle
    return ModelMetadata(
        model_version=bundle.model_version,
        policy_version=bundle.policy_version,
        label_count=len(bundle.classes),
        artifact_schema_version=bundle.artifact_schema_version,
        scope=str(bundle.metrics.get("scope", "unknown")),
        metrics=bundle.metrics,
    )


@app.get("/api/v1/intents", response_model=list[IntentMetadata])
def intents(service: TriageService = Depends(current_service)):
    return [
        IntentMetadata(
            intent=key,
            display_name=str(value.get("display_name", key)),
            queue=str(value.get("queue", "Human Review")),
            mandatory_review=bool(value.get("mandatory_review", True)),
        )
        for key, value in sorted(service.bundle.routes.items())
    ]


@app.post("/api/v1/triage", response_model=TriageResponse)
def triage(
    payload: TriageRequest,
    request: Request,
    _: None = Depends(require_key),
    service: TriageService = Depends(current_service),
):
    if not service.rate_limiter.allow(1):
        raise HTTPException(status_code=429, detail="rate_limit_exceeded")
    try:
        result = service.triage(payload.text, request_id(request), payload.include_examples)
        log_inference(request_id(request), 1, "ok", 1)
        return result
    except RuntimeError as exc:
        if str(exc) == "inference_busy":
            log_inference(request_id(request), 1, "busy")
            raise HTTPException(status_code=429, detail="inference_busy") from exc
        raise


@app.post("/api/v1/triage/batch", response_model=BatchResponse)
def triage_batch(
    payload: BatchRequest,
    request: Request,
    _: None = Depends(require_key),
    service: TriageService = Depends(current_service),
):
    if not service.rate_limiter.allow(len(payload.items)):
        raise HTTPException(status_code=429, detail="rate_limit_exceeded")
    started = time.perf_counter()
    results: list[BatchResult] = []
    valid = 0
    for index, item in enumerate(payload.items):
        try:
            result = service.triage(item.text, f"{request_id(request)}-{index}", False)
            results.append(
                BatchResult(row_index=index, client_id=item.client_id, ok=True, result=result)
            )
            valid += 1
        except (ValueError, RuntimeError) as exc:
            results.append(
                BatchResult(
                    row_index=index,
                    client_id=item.client_id,
                    ok=False,
                    error_code="inference_error",
                    error_message=str(exc),
                )
            )
    log_inference(request_id(request), len(results), "ok", valid)
    return BatchResponse(
        request_id=request_id(request),
        model_version=service.bundle.model_version,
        policy_version=service.bundle.policy_version,
        results=results,
        total=len(results),
        valid=valid,
        invalid=len(results) - valid,
        processing_ms=round((time.perf_counter() - started) * 1000, 3),
    )
