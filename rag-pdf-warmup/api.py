"""HTTP API over rag.answer(): POST /query and GET /health.

Run with: uvicorn api:app --reload
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from functools import cache
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from openai import OpenAIError
from pinecone import PineconeError
from pydantic import BaseModel, Field, StringConstraints

from ingest import ConfigError
from rag import DEFAULT_K, ChatModel, Retriever, answer, build_dependencies

TRACE_HEADER_IN = "X-Request-ID"
TRACE_HEADER_OUT = "X-Trace-ID"
# Only accept caller-supplied trace IDs that are safe to echo and log.
_TRACE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

MAX_K = 20
MAX_QUESTION_LENGTH = 2000

_trace_id: ContextVar[str] = ContextVar("trace_id", default="-")


# --- Structured logging -------------------------------------------------------


class _TraceIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _trace_id.get()
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "event": record.getMessage(),
            "trace_id": getattr(record, "trace_id", "-"),
            **getattr(record, "fields", {}),
        }
        return json.dumps(payload)


logger = logging.getLogger("rag_api")
logger.setLevel(logging.INFO)
logger.addFilter(_TraceIdFilter())
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(_JsonFormatter())
    logger.addHandler(_handler)


# --- Schemas ------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_LENGTH
        ),
    ]
    k: int = Field(default=DEFAULT_K, ge=1, le=MAX_K)


class SourceOut(BaseModel):
    number: int
    source: str
    page: int
    snippet: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ErrorResponse(BaseModel):
    detail: str


# --- App ----------------------------------------------------------------------

app = FastAPI(title="PDF RAG warmup")


@app.middleware("http")
async def trace_requests(request: Request, call_next) -> Response:
    incoming = request.headers.get(TRACE_HEADER_IN, "")
    trace_id = incoming if _TRACE_ID_PATTERN.match(incoming) else uuid.uuid4().hex
    token = _trace_id.set(trace_id)
    try:
        response = await call_next(request)
    finally:
        _trace_id.reset(token)
    response.headers[TRACE_HEADER_OUT] = trace_id
    return response


RagDependencies = tuple[Retriever, ChatModel]


@cache
def _load_rag_dependencies() -> RagDependencies:
    """Build clients once, on first use. A failed build raises and is not cached,
    so fixing .env and retrying works without a restart."""
    return build_dependencies()


def get_rag_dependencies() -> Callable[[], RagDependencies]:
    """Hand the endpoint a loader rather than built clients, so the build runs
    after request validation (an invalid body is a 422, never a 503) and so
    startup and /health never need API keys."""
    return _load_rag_dependencies


@app.exception_handler(ConfigError)
async def _config_error(request: Request, error: ConfigError) -> JSONResponse:
    logger.error("query_failed", extra={"fields": {"error_type": "ConfigError"}})
    return JSONResponse(status_code=503, content={"detail": str(error)})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness only: no external calls, so it stays fast and free."""
    return HealthResponse()


@app.post(
    "/query",
    response_model=QueryResponse,
    responses={502: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def query(
    request: QueryRequest,
    load_dependencies: Annotated[
        Callable[[], RagDependencies], Depends(get_rag_dependencies)
    ],
) -> QueryResponse:
    # Sync `def`: answer() blocks on network I/O, so FastAPI runs this in a
    # threadpool instead of stalling the event loop.
    started = time.perf_counter()
    try:
        retriever, chat_model = load_dependencies()
        result = answer(
            request.question, request.k, retriever=retriever, chat_model=chat_model
        )
    except ConfigError:
        raise
    except (OpenAIError, PineconeError, RuntimeError) as error:
        # Log the full error server-side; return a generic message so upstream
        # details (request IDs, account info) never reach the client.
        logger.error(
            "query_failed",
            extra={"fields": {"error_type": type(error).__name__, "error": str(error)}},
        )
        raise HTTPException(status_code=502, detail="Upstream service error") from error

    logger.info(
        "query_completed",
        extra={
            "fields": {
                "k": request.k,
                "sources": len(result.sources),
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }
        },
    )
    return QueryResponse(
        answer=result.text,
        sources=[
            SourceOut(
                number=s.number,
                source=s.source,
                page=s.page,
                snippet=s.snippet,
                score=s.score,
            )
            for s in result.sources
        ],
    )
