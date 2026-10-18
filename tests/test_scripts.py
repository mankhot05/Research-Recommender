"""Tests for the ingest and search command-line scripts (no network)."""

from __future__ import annotations

import pytest

from app.embeddings import EmbedderConfigError
from app.retrieval import RetrievalConfigError
from scripts import ingest_arxiv
from scripts import search as search_script
from tests.fakes import FakeEmbedder, FakeIndex, make_paper


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch):
    # Keep a developer's real .env out of the tests.
    monkeypatch.setattr(ingest_arxiv, "load_dotenv", lambda: None)
    monkeypatch.setattr(search_script, "load_dotenv", lambda: None)


def test_ingest_fetches_and_upserts(monkeypatch, capsys):
    calls = {}
    papers = [make_paper("2401.00001"), make_paper("2401.00002")]
    index = FakeIndex()

    def fake_fetch(query, max_results):
        calls["fetch"] = (query, max_results)
        return papers

    def fake_upsert(got_papers, embedder, *, index):
        calls["upsert"] = (got_papers, embedder, index)
        return len(got_papers)

    embedder = FakeEmbedder()
    monkeypatch.setattr(ingest_arxiv, "get_embedder", lambda: embedder)
    monkeypatch.setattr(ingest_arxiv, "get_index", lambda e: index)
    monkeypatch.setattr(ingest_arxiv, "fetch_arxiv", fake_fetch)
    monkeypatch.setattr(ingest_arxiv, "upsert_papers", fake_upsert)

    code = ingest_arxiv.main(["--query", "graph neural networks", "--max", "20"])

    assert code == 0
    assert calls["fetch"] == ("graph neural networks", 20)
    assert calls["upsert"] == (papers, embedder, index)
    assert "Upserted 2 papers into papers-fake-3" in capsys.readouterr().out


def test_ingest_defaults_to_200(monkeypatch):
    seen = {}
    monkeypatch.setattr(ingest_arxiv, "get_embedder", FakeEmbedder)
    monkeypatch.setattr(ingest_arxiv, "get_index", lambda e: FakeIndex())
    monkeypatch.setattr(
        ingest_arxiv,
        "fetch_arxiv",
        lambda query, max_results: seen.setdefault("max", max_results) and [],
    )
    monkeypatch.setattr(ingest_arxiv, "upsert_papers", lambda *a, **k: 0)

    assert ingest_arxiv.main(["--query", "q"]) == 0
    assert seen["max"] == 200


def test_ingest_reports_config_errors(monkeypatch, capsys):
    def missing_key(embedder):
        raise RetrievalConfigError("Missing PINECONE_API_KEY. Set it in .env")

    monkeypatch.setattr(ingest_arxiv, "get_embedder", FakeEmbedder)
    monkeypatch.setattr(ingest_arxiv, "get_index", missing_key)

    assert ingest_arxiv.main(["--query", "q"]) == 1
    assert "PINECONE_API_KEY" in capsys.readouterr().err


def test_search_passes_arguments_and_prints_results(monkeypatch, capsys):
    calls = {}
    paper = make_paper("2401.01234", title="Attention", year=2024)

    def fake_search(question, k, year_min, categories):
        calls["search"] = (question, k, year_min, categories)
        return [(paper, 0.8766)]

    monkeypatch.setattr(search_script, "search", fake_search)

    code = search_script.main(
        [
            "message passing",
            "--k",
            "3",
            "--year-min",
            "2022",
            "--category",
            "cs.LG",
            "--category",
            "stat.ML",
        ]
    )

    assert code == 0
    assert calls["search"] == ("message passing", 3, 2022, ["cs.LG", "stat.ML"])
    out = capsys.readouterr().out
    assert "1. [0.877] 2024 cs.LG  Attention" in out
    assert "https://arxiv.org/abs/2401.01234" in out


def test_search_defaults(monkeypatch):
    calls = {}

    def fake_search(question, k, year_min, categories):
        calls["search"] = (question, k, year_min, categories)
        return []

    monkeypatch.setattr(search_script, "search", fake_search)

    assert search_script.main(["q"]) == 0
    assert calls["search"] == ("q", 5, None, None)


def test_search_reports_config_errors(monkeypatch, capsys):
    def missing_key(*args, **kwargs):
        raise EmbedderConfigError("Missing OPENAI_API_KEY. Set it in .env")

    monkeypatch.setattr(search_script, "search", missing_key)

    assert search_script.main(["q"]) == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err
