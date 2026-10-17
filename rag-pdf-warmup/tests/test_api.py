"""API tests with TestClient. answer() is mocked and dependencies are overridden,
so no OpenAI or Pinecone calls happen and no API keys are needed."""

import json
import logging

import pytest
from fastapi.testclient import TestClient
from openai import OpenAIError
from pinecone import PineconeError

import api
from ingest import ConfigError
from rag import Answer, Source

RETRIEVER = object()
CHAT_MODEL = object()

ANSWER = Answer(
    text="Heads run in parallel [1].",
    sources=[
        Source(
            number=1,
            source="Attention Is All You Need.pdf",
            page=4,
            snippet="Multi-head attention runs h attention heads in parallel.",
            score=0.91,
        )
    ],
)


class FakeAnswer:
    """Stands in for rag.answer: records calls, returns or raises a preset result."""

    def __init__(self, result=ANSWER):
        self.result = result
        self.calls = []

    def __call__(self, question, k, *, retriever, chat_model):
        self.calls.append((question, k, retriever, chat_model))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture
def client():
    api.app.dependency_overrides[api.get_rag_dependencies] = lambda: (
        lambda: (
            RETRIEVER,
            CHAT_MODEL,
        )
    )
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


@pytest.fixture
def fake_answer(monkeypatch):
    fake = FakeAnswer()
    monkeypatch.setattr(api, "answer", fake)
    return fake


# --- /health ------------------------------------------------------------------


def test_health_needs_no_dependencies_or_keys():
    # No dependency override: /health must never build the RAG clients.
    response = TestClient(api.app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- /query happy path --------------------------------------------------------


def test_query_returns_answer_and_sources(client, fake_answer):
    response = client.post("/query", json={"question": "  What is MHA?  ", "k": 2})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Heads run in parallel [1].",
        "sources": [
            {
                "number": 1,
                "source": "Attention Is All You Need.pdf",
                "page": 4,
                "snippet": "Multi-head attention runs h attention heads in parallel.",
                "score": 0.91,
            }
        ],
    }
    assert fake_answer.calls == [("What is MHA?", 2, RETRIEVER, CHAT_MODEL)]


def test_query_k_defaults_to_4(client, fake_answer):
    client.post("/query", json={"question": "q"})

    assert fake_answer.calls[0][1] == 4


def test_query_with_no_sources(client, monkeypatch):
    monkeypatch.setattr(api, "answer", FakeAnswer(Answer("I don't know", [])))

    response = client.post("/query", json={"question": "q"})

    assert response.status_code == 200
    assert response.json() == {"answer": "I don't know", "sources": []}


# --- /query unhappy paths -------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 2001},
        {"question": "q", "k": 0},
        {"question": "q", "k": 21},
        {"question": "q", "k": "many"},
    ],
)
def test_invalid_request_is_422_and_never_calls_answer(client, fake_answer, body):
    response = client.post("/query", json=body)

    assert response.status_code == 422
    assert fake_answer.calls == []


@pytest.mark.parametrize(
    "error",
    [
        OpenAIError("429 insufficient_quota for org-secret123"),
        PineconeError("index host unreachable"),
        RuntimeError("Chat model 'x' returned an empty answer"),
    ],
)
def test_upstream_failure_is_502_with_generic_detail(client, monkeypatch, error):
    monkeypatch.setattr(api, "answer", FakeAnswer(error))

    response = client.post("/query", json={"question": "q"})

    assert response.status_code == 502
    assert response.json() == {"detail": "Upstream service error"}


def test_missing_config_is_503_with_message(client, monkeypatch):
    monkeypatch.setattr(
        api,
        "answer",
        FakeAnswer(ConfigError("Missing OPENAI_CHAT_MODEL. Set it in .env")),
    )

    response = client.post("/query", json={"question": "q"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Missing OPENAI_CHAT_MODEL. Set it in .env"}


def test_dependency_build_failure_is_503(monkeypatch, fake_answer):
    def missing_key():
        raise ConfigError("Missing PINECONE_API_KEY. Set it in .env")

    api.app.dependency_overrides[api.get_rag_dependencies] = lambda: missing_key
    try:
        client = TestClient(api.app)
        response = client.post("/query", json={"question": "q"})
        invalid = client.post("/query", json={"question": "", "k": 0})
    finally:
        api.app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "PINECONE_API_KEY" in response.json()["detail"]
    # Validation runs before clients are built: a bad body is 422, not 503.
    assert invalid.status_code == 422
    assert fake_answer.calls == []


# --- tracing and structured logs ------------------------------------------------


def test_incoming_request_id_is_echoed_as_trace_id(client, fake_answer):
    response = client.post(
        "/query", json={"question": "q"}, headers={"X-Request-ID": "abc-123"}
    )

    assert response.headers["X-Trace-ID"] == "abc-123"


@pytest.mark.parametrize("header", [None, "bad id\nwith newline", "x" * 200])
def test_trace_id_is_generated_when_missing_or_unsafe(client, header):
    headers = {"X-Request-ID": header} if header else {}

    response = client.get("/health", headers=headers)

    trace_id = response.headers["X-Trace-ID"]
    assert trace_id != header
    assert len(trace_id) == 32
    assert trace_id.isalnum()


def test_query_logs_carry_trace_id(client, fake_answer, caplog):
    with caplog.at_level(logging.INFO, logger="rag_api"):
        client.post("/query", json={"question": "q"}, headers={"X-Request-ID": "t-1"})

    [record] = [r for r in caplog.records if r.getMessage() == "query_completed"]
    assert record.trace_id == "t-1"
    assert record.fields["k"] == 4
    assert record.fields["sources"] == 1

    line = json.loads(api._JsonFormatter().format(record))
    assert line["event"] == "query_completed"
    assert line["trace_id"] == "t-1"
    assert "latency_ms" in line


def test_failure_logs_error_type_server_side(client, monkeypatch, caplog):
    monkeypatch.setattr(api, "answer", FakeAnswer(OpenAIError("quota exceeded")))

    with caplog.at_level(logging.ERROR, logger="rag_api"):
        client.post("/query", json={"question": "q"}, headers={"X-Request-ID": "t-2"})

    [record] = [r for r in caplog.records if r.getMessage() == "query_failed"]
    assert record.trace_id == "t-2"
    assert record.fields["error_type"] == "OpenAIError"
    assert record.fields["error"] == "quota exceeded"
