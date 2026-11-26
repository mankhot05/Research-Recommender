"""Tests for the ingest, search and export command-line scripts (no network)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from app.embeddings import EmbedderConfigError
from app.graph import GraphDeps
from app.llm import LLMConfigError
from app.retrieval import RetrievalConfigError, paper_to_record
from app.sources import append_to_cache, load_cached_papers
from scripts import draw_graph, export_cache, ingest_arxiv, recommend
from scripts import search as search_script
from tests.fakes import (
    FakeArxiv,
    FakeCritic,
    FakeEmbedder,
    FakeIndex,
    FakePaperStore,
    FakeRewriter,
    make_paper,
)


@pytest.fixture(autouse=True)
def temp_cache(monkeypatch, tmp_path):
    # Keep tests away from the real data/papers.jsonl.
    path = tmp_path / "data" / "papers.jsonl"
    monkeypatch.setattr(ingest_arxiv, "CACHE_PATH", path)
    monkeypatch.setattr(export_cache, "CACHE_PATH", path)
    return path


def no_fetch(query, max_results):
    raise AssertionError("arXiv must not be called")


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
    monkeypatch.setattr(ingest_arxiv, "get_embedder", lambda name: embedder)
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
    monkeypatch.setattr(ingest_arxiv, "get_embedder", lambda name: FakeEmbedder())
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

    monkeypatch.setattr(ingest_arxiv, "get_embedder", lambda name: FakeEmbedder())
    monkeypatch.setattr(ingest_arxiv, "get_index", missing_key)

    assert ingest_arxiv.main(["--query", "q"]) == 1
    assert "PINECONE_API_KEY" in capsys.readouterr().err


def test_ingest_query_appends_to_cache(monkeypatch, temp_cache):
    papers = [make_paper("2401.00001"), make_paper("2401.00002")]
    monkeypatch.setattr(ingest_arxiv, "get_embedder", lambda name: FakeEmbedder())
    monkeypatch.setattr(ingest_arxiv, "get_index", lambda e: FakeIndex())
    monkeypatch.setattr(ingest_arxiv, "fetch_arxiv", lambda query, max_results: papers)
    monkeypatch.setattr(ingest_arxiv, "upsert_papers", lambda *a, **k: len(papers))

    assert ingest_arxiv.main(["--query", "q"]) == 0
    assert load_cached_papers(temp_cache) == papers


def test_ingest_caches_before_upsert_fails(monkeypatch, temp_cache):
    papers = [make_paper("2401.00001")]

    def failing_upsert(*args, **kwargs):
        raise ValueError("upsert failed")

    monkeypatch.setattr(ingest_arxiv, "get_embedder", lambda name: FakeEmbedder())
    monkeypatch.setattr(ingest_arxiv, "get_index", lambda e: FakeIndex())
    monkeypatch.setattr(ingest_arxiv, "fetch_arxiv", lambda query, max_results: papers)
    monkeypatch.setattr(ingest_arxiv, "upsert_papers", failing_upsert)

    assert ingest_arxiv.main(["--query", "q"]) == 1
    assert load_cached_papers(temp_cache) == papers


def test_ingest_from_cache_skips_arxiv(monkeypatch, capsys, temp_cache):
    papers = [make_paper("2401.00001"), make_paper("2401.00002")]
    append_to_cache(papers, temp_cache)
    upserted = {}

    def fake_upsert(got_papers, embedder, *, index):
        upserted["papers"] = got_papers
        return len(got_papers)

    monkeypatch.setattr(ingest_arxiv, "get_embedder", lambda name: FakeEmbedder())
    monkeypatch.setattr(ingest_arxiv, "get_index", lambda e: FakeIndex())
    monkeypatch.setattr(ingest_arxiv, "fetch_arxiv", no_fetch)
    monkeypatch.setattr(ingest_arxiv, "upsert_papers", fake_upsert)

    assert ingest_arxiv.main(["--from-cache"]) == 0
    assert upserted["papers"] == papers
    out = capsys.readouterr().out
    assert "Loaded 2 papers" in out
    assert "Upserted 2 papers into papers-fake-3" in out


def test_ingest_from_empty_cache_fails(monkeypatch, capsys):
    monkeypatch.setattr(ingest_arxiv, "get_embedder", lambda name: FakeEmbedder())
    monkeypatch.setattr(ingest_arxiv, "get_index", lambda e: FakeIndex())
    monkeypatch.setattr(ingest_arxiv, "fetch_arxiv", no_fetch)

    assert ingest_arxiv.main(["--from-cache"]) == 1
    assert "no cached papers" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "expected"),
    [(["--query", "q"], None), (["--query", "q", "--embedder", "specter"], "specter")],
)
def test_ingest_passes_embedder_override(monkeypatch, argv, expected):
    seen = {}

    def fake_get_embedder(name):
        seen["name"] = name
        return FakeEmbedder()

    monkeypatch.setattr(ingest_arxiv, "get_embedder", fake_get_embedder)
    monkeypatch.setattr(ingest_arxiv, "get_index", lambda e: FakeIndex())
    monkeypatch.setattr(ingest_arxiv, "fetch_arxiv", lambda query, max_results: [])
    monkeypatch.setattr(ingest_arxiv, "upsert_papers", lambda *a, **k: 0)

    assert ingest_arxiv.main(argv) == 0
    assert seen["name"] == expected


@pytest.mark.parametrize("argv", [[], ["--query", "q", "--from-cache"]])
def test_ingest_needs_exactly_one_source(argv):
    with pytest.raises(SystemExit) as excinfo:
        ingest_arxiv.main(argv)
    assert excinfo.value.code == 2


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


def index_holding(*papers):
    return FakeIndex(
        records={p.arxiv_id: paper_to_record(p, [0.0])["metadata"] for p in papers}
    )


def test_export_writes_index_papers_to_cache(monkeypatch, capsys, temp_cache):
    papers = [make_paper("2401.00001"), make_paper("2401.00002")]
    opened = []

    def fake_open_index(name):
        opened.append(name)
        return index_holding(*papers)

    monkeypatch.setattr(export_cache, "open_index", fake_open_index)

    assert export_cache.main([]) == 0
    assert opened == ["papers-openai-1536"]
    assert load_cached_papers(temp_cache) == papers
    assert "Exported 2 papers from papers-openai-1536; cached 2 new" in (
        capsys.readouterr().out
    )


def test_export_merges_with_existing_cache(monkeypatch, capsys, temp_cache):
    old, shared, new = (make_paper(f"2401.0000{i}") for i in range(1, 4))
    append_to_cache([old, shared], temp_cache)
    monkeypatch.setattr(
        export_cache, "open_index", lambda name: index_holding(shared, new)
    )

    assert export_cache.main(["--index", "papers-specter-768"]) == 0
    assert load_cached_papers(temp_cache) == [old, shared, new]
    assert len(temp_cache.read_text().splitlines()) == 3
    assert "Exported 2 papers from papers-specter-768; cached 1 new" in (
        capsys.readouterr().out
    )


def test_export_reports_missing_index(monkeypatch, capsys, temp_cache):
    def missing(name):
        raise RetrievalConfigError(f"Index {name!r} does not exist")

    monkeypatch.setattr(export_cache, "open_index", missing)

    assert export_cache.main([]) == 1
    assert "does not exist" in capsys.readouterr().err
    assert not temp_cache.exists()


@pytest.mark.parametrize(
    "script",
    [
        "ingest_arxiv.py",
        "search.py",
        "compare_embedders.py",
        "export_cache.py",
        "recommend.py",
        "draw_graph.py",
    ],
)
def test_scripts_run_as_files_from_repo_root(script):
    # Needs the package installed (pip install -e .), so `import app` resolves.
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, f"scripts/{script}", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def fake_graph_deps(critic: FakeCritic) -> tuple[GraphDeps, FakeArxiv]:
    """Deps for one weak pass (1 relevant) then a good one (6 relevant)."""
    arxiv = FakeArxiv(
        {
            "query 1": [
                make_paper(f"2401.{n:05d}", title=f"Paper {n}") for n in (1, 2)
            ],
            "query 2": [
                make_paper(f"2401.{n:05d}", title=f"Paper {n}") for n in range(3, 10)
            ],
        }
    )
    store = FakePaperStore()
    deps = GraphDeps(
        rewrite_queries=FakeRewriter(),
        critique=critic,
        fetch_arxiv=arxiv,
        embedder=FakeEmbedder(),
        upsert_papers=store.upsert,
        search=store.search,
    )
    return deps, arxiv


def relevant_ids(*numbers: int) -> set[str]:
    return {f"2401.{n:05d}" for n in numbers}


def test_recommend_prints_iterations_and_top_papers(monkeypatch, capsys):
    critic = FakeCritic([relevant_ids(1), relevant_ids(1, 3, 4, 5, 8, 9)])
    deps, _ = fake_graph_deps(critic)
    seen = {}

    def fake_default_deps(*, fetch):
        seen["fetch"] = fetch
        return deps

    monkeypatch.setattr(recommend, "default_deps", fake_default_deps)

    code = recommend.main(["GNNs for molecules"])

    out = capsys.readouterr().out
    assert code == 0
    assert seen["fetch"] is True
    assert "Iteration 1: 1 of 2 relevant\n  - query 1\n" in out
    assert "Iteration 2: 6 of 9 relevant\n  - query 2\n" in out
    assert "Top 5:" in out
    assert (
        "1. Paper 9 (2024)\n   https://arxiv.org/abs/2401.00009\n   reason 2401.00009\n"
    ) in out
    assert "5. Paper 3 (2024)" in out  # relevant by score: 9, 8, 5, 4, 3
    assert "6. " not in out


def test_recommend_no_fetch_skips_arxiv_and_runs_once(monkeypatch, capsys):
    deps, _ = fake_graph_deps(FakeCritic([set()]))
    seen = {}

    def fake_default_deps(*, fetch):
        seen["fetch"] = fetch
        return deps

    monkeypatch.setattr(recommend, "default_deps", fake_default_deps)

    code = recommend.main(["GNNs for molecules", "--no-fetch"])

    out = capsys.readouterr().out
    assert code == 0
    assert seen["fetch"] is False
    assert "running 1 iteration" in out
    assert "Iteration 2" not in out
    assert "No relevant papers found." in out


def test_recommend_reports_config_errors(monkeypatch, capsys):
    def broken(*, fetch):
        raise LLMConfigError("Missing OPENAI_API_KEY. Set it in .env")

    monkeypatch.setattr(recommend, "default_deps", broken)

    assert recommend.main(["question"]) == 1
    assert "Missing OPENAI_API_KEY" in capsys.readouterr().err


def test_draw_graph_writes_mermaid(tmp_path, capsys):
    output = tmp_path / "graph.md"

    assert draw_graph.main(["--output", str(output)]) == 0

    text = output.read_text()
    assert "```mermaid" in text
    for node in (
        "rewrite_query",
        "fetch_candidates",
        "index_candidates",
        "retrieve",
        "critique",
        "finalize",
    ):
        assert node in text


def test_committed_diagrams_are_up_to_date():
    # If this fails after changing app/graph.py, run: python scripts/draw_graph.py
    # and paste the new mermaid block into README.md.
    root = Path(__file__).resolve().parent.parent
    expected = draw_graph.render()
    assert (root / "docs" / "graph.md").read_text() == expected
    mermaid_block = expected[expected.index("```mermaid") :]
    assert mermaid_block in (root / "README.md").read_text()
