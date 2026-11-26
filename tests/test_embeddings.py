"""Tests for app.embeddings using fake embedders and clients (no network)."""

from __future__ import annotations

import sys
from datetime import date
from types import SimpleNamespace

import pytest
from langchain_openai import OpenAIEmbeddings

from app import embeddings
from app.embeddings import (
    Embedder,
    EmbedderConfigError,
    OpenAIEmbedder,
    SciNCLEmbedder,
    SentenceTransformerEmbedder,
    SpecterEmbedder,
    get_embedder,
)
from app.retrieval import index_name
from app.schemas import Paper
from tests.fakes import FakeEmbedder, FakeSentenceTransformer, make_paper


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


def test_openai_formats_paper_as_title_blank_line_abstract():
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
    embedder = OpenAIEmbedder(client=FakeLangChainClient())

    assert embedder.format_paper(paper) == (
        "Attention Is All You Need\n\nWe propose a new architecture."
    )


def test_implementations_satisfy_protocol():
    assert isinstance(FakeEmbedder(), Embedder)
    assert isinstance(OpenAIEmbedder(client=FakeLangChainClient()), Embedder)
    assert isinstance(SciNCLEmbedder(model=FakeSentenceTransformer()), Embedder)
    assert isinstance(SpecterEmbedder(model=FakeSentenceTransformer()), Embedder)


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


def test_get_embedder_name_overrides_setting(clean_env):
    clean_env.setitem(embeddings._EMBEDDERS, "fake", FakeEmbedder)
    clean_env.setenv("EMBEDDER", "openai")

    assert isinstance(get_embedder(" Fake "), FakeEmbedder)


def test_get_embedder_rejects_unknown_override(clean_env):
    with pytest.raises(EmbedderConfigError, match="Unknown EMBEDDER 'bogus'"):
        get_embedder("bogus")


# --- local sentence-transformers models (SciNCL, SPECTER) -------------------

LOCAL_EMBEDDERS = [
    (SciNCLEmbedder, "scincl", "malteos/scincl"),
    (SpecterEmbedder, "specter", "sentence-transformers/allenai-specter"),
]


@pytest.fixture
def no_model_loading(monkeypatch):
    """Fail the test if anything tries to load (download) a real model."""

    def refuse(model_id):
        raise AssertionError(f"tried to load {model_id}")

    monkeypatch.setattr(embeddings, "_load_sentence_transformer", refuse)


@pytest.mark.parametrize(("cls", "name", "model_id"), LOCAL_EMBEDDERS)
def test_local_embedder_metadata(cls, name, model_id):
    assert cls.name == name
    assert cls.model_id == model_id
    assert cls.dimension == 768
    assert index_name(cls(model=FakeSentenceTransformer())) == f"papers-{name}-768"


@pytest.mark.parametrize(("cls", "name", "model_id"), LOCAL_EMBEDDERS)
def test_get_embedder_builds_local_models_without_loading(
    clean_env, no_model_loading, cls, name, model_id
):
    assert isinstance(get_embedder(name), cls)
    clean_env.setenv("EMBEDDER", name)
    assert isinstance(get_embedder(), cls)


@pytest.mark.parametrize("cls", [SciNCLEmbedder, SpecterEmbedder])
def test_local_embedder_formats_paper_with_sep_token(cls):
    paper = make_paper(title="Title", abstract="Abstract.")

    assert cls(model=FakeSentenceTransformer()).format_paper(paper) == (
        "Title[SEP]Abstract."
    )


def test_local_embed_documents_batches_and_normalizes():
    model = FakeSentenceTransformer()
    embedder = SciNCLEmbedder(model=model)

    vectors = embedder.embed_documents(["ab", "abcd"])

    assert [v[0] for v in vectors] == [2.0, 4.0]
    assert all(type(v) is list and len(v) == 768 for v in vectors)
    assert all(type(x) is float for x in vectors[0])
    [(texts, kwargs)] = model.calls
    assert texts == ["ab", "abcd"]
    assert kwargs["batch_size"] == 16
    assert kwargs["normalize_embeddings"] is True


def test_local_embed_query_returns_one_vector():
    model = FakeSentenceTransformer()

    vector = SpecterEmbedder(model=model).embed_query("abc")

    assert len(vector) == 768
    assert vector[0] == 3.0
    assert model.calls[0][0] == ["abc"]


def test_local_embed_documents_empty_input_skips_model():
    model = FakeSentenceTransformer()

    assert SciNCLEmbedder(model=model).embed_documents([]) == []
    assert model.calls == []


@pytest.mark.parametrize("method", ["embed_documents", "embed_query"])
def test_local_wrong_dimension_raises(method):
    embedder = SciNCLEmbedder(model=FakeSentenceTransformer(dimension=384))
    arg = ["text"] if method == "embed_documents" else "text"

    with pytest.raises(RuntimeError, match="384-d vector, expected 768"):
        getattr(embedder, method)(arg)


def test_local_model_loads_lazily_on_first_use(monkeypatch):
    loaded = []

    def fake_load(model_id):
        loaded.append(model_id)
        return FakeSentenceTransformer()

    monkeypatch.setattr(embeddings, "_load_sentence_transformer", fake_load)
    embedder = SciNCLEmbedder()
    assert loaded == []

    embedder.embed_query("a")
    embedder.embed_documents(["b", "c"])

    assert loaded == ["malteos/scincl"]


def test_load_sentence_transformer_builds_each_model_once(monkeypatch):
    built = []

    class CountingSentenceTransformer(FakeSentenceTransformer):
        def __init__(self, model_id, device):
            super().__init__()
            built.append((model_id, device))

    # A stub module, so the test neither imports the real (slow) library nor downloads.
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=CountingSentenceTransformer),
    )
    monkeypatch.setattr(embeddings, "_device", lambda: "cpu")
    embeddings._load_sentence_transformer.cache_clear()
    try:
        first = embeddings._load_sentence_transformer("malteos/scincl")
        again = embeddings._load_sentence_transformer("malteos/scincl")
        other = embeddings._load_sentence_transformer(
            "sentence-transformers/allenai-specter"
        )
    finally:
        embeddings._load_sentence_transformer.cache_clear()

    assert first is again
    assert other is not first
    assert built == [
        ("malteos/scincl", "cpu"),
        ("sentence-transformers/allenai-specter", "cpu"),
    ]


@pytest.mark.parametrize(("available", "device"), [(True, "mps"), (False, "cpu")])
def test_device_prefers_apple_mps(monkeypatch, available, device):
    import torch

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: available)

    assert embeddings._device() == device


def test_shared_base_is_not_registered():
    assert SentenceTransformerEmbedder not in embeddings._EMBEDDERS.values()
