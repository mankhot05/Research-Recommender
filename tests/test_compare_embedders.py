"""Tests for scripts/compare_embedders.py with fake embedders and indexes (no network)."""

from __future__ import annotations

import pytest

from app.retrieval import RetrievalConfigError
from scripts import compare_embedders
from scripts.compare_embedders import format_report, load_queries, run_comparison
from tests.fakes import FakeEmbedder, FakeIndex, FakeMatch, make_paper


def fake_index(*hits: tuple[str, str, float]) -> FakeIndex:
    """A FakeIndex returning (arxiv_id, title, score) matches in the given order."""
    matches = [
        FakeMatch(
            id=arxiv_id,
            score=score,
            metadata=make_paper(arxiv_id, title=title).model_dump(mode="json"),
        )
        for arxiv_id, title, score in hits
    ]
    return FakeIndex(matches)


class FakeSetup:
    """Stands in for get_embedder/get_index: one named fake embedder and index each."""

    def __init__(self, indexes: dict[str, FakeIndex]) -> None:
        self.indexes = indexes
        self.built: list[str] = []
        self.index_calls: list[tuple[str, bool]] = []

    def get_embedder(self, name: str) -> FakeEmbedder:
        self.built.append(name)
        embedder = FakeEmbedder()
        embedder.name = name
        return embedder

    def get_index(self, embedder: FakeEmbedder, *, create: bool = True) -> FakeIndex:
        self.index_calls.append((embedder.name, create))
        return self.indexes[embedder.name]


@pytest.fixture
def setup(monkeypatch):
    fake = FakeSetup(
        {
            "openai": fake_index(
                ("2401.00001", "GNN A", 0.81), ("2401.00002", "GNN B", 0.7)
            ),
            "scincl": fake_index(("2401.00002", "GNN B", 0.93)),
        }
    )
    monkeypatch.setattr(compare_embedders, "get_embedder", fake.get_embedder)
    monkeypatch.setattr(compare_embedders, "get_index", fake.get_index)
    return fake


def test_load_queries_skips_blanks_and_comments(tmp_path):
    path = tmp_path / "queries.txt"
    path.write_text(
        "# header\n\n  graph neural networks  \n# another\nprotein folding\n"
    )

    assert load_queries(path) == ["graph neural networks", "protein folding"]


def test_load_queries_rejects_empty_file(tmp_path):
    path = tmp_path / "queries.txt"
    path.write_text("# only comments\n\n")

    with pytest.raises(ValueError, match="no queries"):
        load_queries(path)


def test_run_comparison_sets_up_each_embedder_once(setup):
    results = run_comparison(["q1", "q2"], ["openai", "scincl"], k=2)

    assert setup.built == ["openai", "scincl"]
    assert setup.index_calls == [("openai", False), ("scincl", False)]
    assert list(results) == ["q1", "q2"]
    assert [p.title for p, _ in results["q1"]["openai"]] == ["GNN A", "GNN B"]
    assert [(p.title, s) for p, s in results["q2"]["scincl"]] == [("GNN B", 0.93)]
    queries = setup.indexes["openai"].queries
    assert [q["top_k"] for q in queries] == [2, 2]


def test_format_report_one_table_per_query():
    a = make_paper("2401.00001", title="Alpha")
    b = make_paper("2401.00002", title="Beta")
    results = {
        "first query": {"scincl": [(b, 0.9)], "openai": [(a, 0.81234), (b, 0.5)]},
        "second query": {"scincl": [], "openai": [(b, 0.4)]},
    }

    report = format_report(results, ["openai", "scincl"], k=2)

    assert "don't compare them across columns" in report
    assert "## 1. first query" in report
    assert "## 2. second query" in report
    assert report.count("| Rank | openai | scincl |") == 2
    assert "| 1 | Alpha (0.812) | Beta (0.900) |" in report
    assert "| 2 | Beta (0.500) | — |" in report
    assert "| 1 | Beta (0.400) | — |" in report
    assert "| 2 | — | — |" in report


def test_format_report_escapes_pipes_in_titles():
    paper = make_paper(title="A | B")

    report = format_report({"q": {"openai": [(paper, 0.5)]}}, ["openai"], k=1)

    assert "| 1 | A \\| B (0.500) |" in report


def test_main_with_queries_writes_report(setup, tmp_path, capsys):
    out = tmp_path / "docs" / "comparison.md"

    code = compare_embedders.main(
        ["graph nets", "--k", "1", "--embedder", "openai", "--embedder", "scincl"]
        + ["--out", str(out)]
    )

    assert code == 0
    report = out.read_text()
    assert "| 1 | GNN A (0.810) | GNN B (0.930) |" in report
    printed = capsys.readouterr().out
    assert "## 1. graph nets" in printed
    assert f"Wrote {out}" in printed


def test_main_reads_queries_file_by_default(setup, tmp_path):
    queries = tmp_path / "queries.txt"
    queries.write_text("first\nsecond\n")
    out = tmp_path / "out.md"

    code = compare_embedders.main(
        ["--queries-file", str(queries), "--embedder", "openai", "--out", str(out)]
    )

    assert code == 0
    report = out.read_text()
    assert "## 1. first" in report
    assert "## 2. second" in report


def test_main_defaults_to_all_three_embedders(monkeypatch, tmp_path):
    fake = FakeSetup({name: fake_index() for name in ("openai", "scincl", "specter")})
    monkeypatch.setattr(compare_embedders, "get_embedder", fake.get_embedder)
    monkeypatch.setattr(compare_embedders, "get_index", fake.get_index)

    assert compare_embedders.main(["q", "--out", str(tmp_path / "out.md")]) == 0
    assert fake.built == ["openai", "scincl", "specter"]


def test_main_missing_index_fails_without_writing(monkeypatch, tmp_path, capsys):
    def missing(embedder, *, create=True):
        raise RetrievalConfigError("Index 'papers-scincl-768' does not exist.")

    monkeypatch.setattr(compare_embedders, "get_embedder", FakeSetup({}).get_embedder)
    monkeypatch.setattr(compare_embedders, "get_index", missing)
    out = tmp_path / "out.md"

    assert compare_embedders.main(["q", "--out", str(out)]) == 1
    assert not out.exists()
    assert "does not exist" in capsys.readouterr().err
