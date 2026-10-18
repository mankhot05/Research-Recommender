"""Tests for app.embeddings using fake embedders and clients (no network)."""

from __future__ import annotations

from datetime import date

import pytest
from langchain_openai import OpenAIEmbeddings

from app import embeddings
from app.embeddings import (
    Embedder,
    EmbedderConfigError,
    OpenAIEmbedder,
    get_embedder,
    paper_to_text,
)
from app.schemas import Paper
from tests.fakes import FakeEmbedder


class FakeLangChainClient:
    """Stands in for OpenAIEmbeddings: returns vectors of a chosen length."""

    def __init__(self, dimension: int = 1536, drop_last: bool = False) -> None:
        self._dimension = dimension
        self._drop_last = drop_last
        self.calls: list[object] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        vectors = [self._vector(i) for i in range(len(texts))]
        return vectors[:-1] if self._drop_last else vectors

    def embed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        return self._vector(0)

    def _vector(self, i: int) -> list[float]:
        return [float(i)] * self._dimension


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("EMBEDDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return monkeypatch


def test_paper_to_text_joins_title_and_abstract():
    paper = Paper(
        arxiv_id="2401.01234",
        title="Attention Is All You Need",
        abstract="We propose a new architecture.",
        authors=["Ada Lovelace"],
        published=date(2024, 1, 3),
        year=2024,
        primary_category="cs.LG",
        url="https://arxiv.org/abs/2401.01234",
    )

    assert paper_to_text(paper) == (
        "Attention Is All You Need\n\nWe propose a new architecture."
    )


def test_implementations_satisfy_protocol():
    assert isinstance(FakeEmbedder(), Embedder)
    assert isinstance(OpenAIEmbedder(client=FakeLangChainClient()), Embedder)


def test_openai_embedder_metadata():
    embedder = OpenAIEmbedder(client=FakeLangChainClient())

    assert embedder.name == "openai"
    assert embedder.model == "text-embedding-3-small"
    assert embedder.dimension == 1536


def test_openai_embedder_delegates_to_client():
    client = FakeLangChainClient()
    embedder = OpenAIEmbedder(client=client)

    vectors = embedder.embed_documents(["a", "b"])
    query_vector = embedder.embed_query("q")

    assert [v[0] for v in vectors] == [0.0, 1.0]
    assert all(len(v) == 1536 for v in vectors)
    assert len(query_vector) == 1536
    assert client.calls == [["a", "b"], "q"]


def test_embed_documents_empty_input_skips_client():
    client = FakeLangChainClient()

    assert OpenAIEmbedder(client=client).embed_documents([]) == []
    assert client.calls == []


@pytest.mark.parametrize("method", ["embed_documents", "embed_query"])
def test_wrong_dimension_raises(method):
    embedder = OpenAIEmbedder(client=FakeLangChainClient(dimension=768))
    arg = ["a"] if method == "embed_documents" else "a"

    with pytest.raises(RuntimeError, match="768-d vector, expected 1536"):
        getattr(embedder, method)(arg)


def test_wrong_vector_count_raises():
    embedder = OpenAIEmbedder(client=FakeLangChainClient(drop_last=True))

    with pytest.raises(RuntimeError, match="1 embeddings for 2 texts"):
        embedder.embed_documents(["a", "b"])


def test_missing_api_key_raises(clean_env):
    with pytest.raises(EmbedderConfigError, match="OPENAI_API_KEY"):
        OpenAIEmbedder()


def test_builds_langchain_client_from_env(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "sk-test")

    client = OpenAIEmbedder()._client

    assert isinstance(client, OpenAIEmbeddings)
    assert client.model == "text-embedding-3-small"
    assert client.dimensions == 1536


@pytest.mark.parametrize("value", [None, "", "openai", " OpenAI "])
def test_get_embedder_defaults_to_openai(clean_env, value):
    clean_env.setenv("OPENAI_API_KEY", "sk-test")
    if value is not None:
        clean_env.setenv("EMBEDDER", value)

    assert isinstance(get_embedder(), OpenAIEmbedder)


def test_get_embedder_rejects_unknown_name(clean_env):
    clean_env.setenv("EMBEDDER", "bogus")

    with pytest.raises(EmbedderConfigError, match="Unknown EMBEDDER 'bogus'"):
        get_embedder()


def test_get_embedder_uses_registered_implementation(clean_env):
    clean_env.setitem(embeddings._EMBEDDERS, "fake", FakeEmbedder)
    clean_env.setenv("EMBEDDER", "fake")

    embedder = get_embedder()

    assert isinstance(embedder, FakeEmbedder)
    assert embedder.embed_query("abcd") == [4.0, 0.0, 1.0]
