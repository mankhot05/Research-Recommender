"""Tests for the LangGraph agent loop, with scripted fakes (no network)."""

from __future__ import annotations

import logging

import pytest

from app import graph
from app.graph import GraphDeps, build_graph, recursion_limit, route_after_critique
from app.schemas import GradedPaper, Paper
from tests.fakes import (
    FakeArxiv,
    FakeCritic,
    FakeEmbedder,
    FakePaperStore,
    FakeRewriter,
    make_paper,
)

QUESTION = "message passing GNNs for molecules"


def paper(n: int) -> Paper:
    """Paper number n; the fake index scores it n / 100, so higher n ranks higher."""
    return make_paper(f"2401.{n:05d}", title=f"Paper {n}")


def ids(*numbers: int) -> set[str]:
    return {f"2401.{n:05d}" for n in numbers}


def make_deps(
    arxiv: FakeArxiv, store: FakePaperStore, critic: FakeCritic
) -> tuple[GraphDeps, FakeRewriter]:
    rewriter = FakeRewriter()
    deps = GraphDeps(
        rewrite_queries=rewriter,
        critique=critic,
        fetch_arxiv=arxiv,
        embedder=FakeEmbedder(),
        upsert_papers=store.upsert,
        search=store.search,
    )
    return deps, rewriter


def test_good_first_pass_finalizes_after_one_iteration():
    arxiv = FakeArxiv({"query 1": [paper(n) for n in range(1, 11)]})
    store = FakePaperStore()
    critic = FakeCritic([ids(2, 4, 6, 7, 8, 9)])
    deps, rewriter = make_deps(arxiv, store, critic)

    state = graph.run_graph(QUESTION, deps=deps)

    assert state["iteration"] == 1
    assert rewriter.calls == [(QUESTION, None)]
    assert arxiv.calls == [("query 1", graph.FETCH_PER_QUERY)]
    # Top 5 relevant, ordered by vector score, each with its own reason.
    final = state["final"]
    assert [r.paper.arxiv_id for r in final] == [
        "2401.00009",
        "2401.00008",
        "2401.00007",
        "2401.00006",
        "2401.00004",
    ]
    assert final[0].score == pytest.approx(0.09)
    assert final[0].reason == "reason 2401.00009"


def test_weak_then_good_loops_once_with_feedback():
    arxiv = FakeArxiv(
        {
            "query 1": [paper(n) for n in range(1, 4)],
            "query 2": [paper(n) for n in range(4, 11)],
        }
    )
    store = FakePaperStore()
    critic = FakeCritic([ids(1, 2), ids(1, 2, 5, 6, 7, 8)])
    deps, rewriter = make_deps(arxiv, store, critic)

    state = graph.run_graph(QUESTION, deps=deps)

    assert state["iteration"] == 2
    # The second rewrite is steered by the first critique's feedback.
    assert rewriter.calls == [(QUESTION, None), (QUESTION, "feedback 0")]
    assert store.upserts == [
        [p.arxiv_id for p in arxiv.by_query["query 1"]],
        [p.arxiv_id for p in arxiv.by_query["query 2"]],
    ]
    assert len(critic.calls) == 2
    assert len(state["final"]) == 5


def test_always_weak_stops_at_max_iterations():
    arxiv = FakeArxiv({f"query {i}": [paper(i)] for i in range(1, 10)})
    store = FakePaperStore()
    critic = FakeCritic([ids(1)])
    deps, rewriter = make_deps(arxiv, store, critic)

    state = graph.run_graph(QUESTION, deps=deps)

    assert state["iteration"] == graph.DEFAULT_MAX_ITERATIONS == 3
    assert len(rewriter.calls) == 3
    # Best effort: whatever relevant papers were found, even fewer than 5.
    assert [r.paper.arxiv_id for r in state["final"]] == ["2401.00001"]


def test_max_iterations_one_runs_a_single_pass():
    deps, rewriter = make_deps(FakeArxiv({}), FakePaperStore(), FakeCritic([set()]))

    result = graph.run_agent(QUESTION, deps=deps, max_iterations=1)

    assert result == []
    assert len(rewriter.calls) == 1


def test_arxiv_failure_still_returns_results_from_the_index(caplog):
    arxiv = FakeArxiv({}, fail=True)
    store = FakePaperStore([paper(n) for n in range(1, 8)])
    critic = FakeCritic([ids(*range(1, 8))])
    deps, _ = make_deps(arxiv, store, critic)

    with caplog.at_level(logging.WARNING, logger="app.graph"):
        result = graph.run_agent(QUESTION, deps=deps)

    assert [r.paper.arxiv_id for r in result] == [
        f"2401.{n:05d}" for n in (7, 6, 5, 4, 3)
    ]
    assert store.upserts == []  # nothing new to index
    assert "continuing with the existing index" in caplog.text


def test_retrieve_always_uses_the_original_question_and_k_10():
    arxiv = FakeArxiv({})
    store = FakePaperStore([paper(1)])
    deps, _ = make_deps(arxiv, store, FakeCritic([set()]))

    graph.run_graph(QUESTION, deps=deps)

    assert store.searches == [(QUESTION, graph.RETRIEVE_K)] * 3
    assert graph.RETRIEVE_K == 10


def graded(relevant_count: int, total: int = 10) -> list[GradedPaper]:
    return [
        GradedPaper(paper=paper(n), score=0.5, relevant=n <= relevant_count, reason="r")
        for n in range(1, total + 1)
    ]


@pytest.mark.parametrize(
    ("relevant", "iteration", "expected"),
    [
        (5, 1, "finalize"),
        (4, 1, "rewrite_query"),
        (4, 3, "finalize"),
        (0, 2, "rewrite_query"),
    ],
)
def test_route_after_critique(relevant, iteration, expected):
    state = {"graded": graded(relevant), "iteration": iteration, "max_iterations": 3}

    assert route_after_critique(state) == expected


def test_recursion_limit_covers_a_full_run():
    # Each loop runs 5 nodes; a full run is 5 * max_iterations + finalize.
    assert recursion_limit(3) > graph.NODES_PER_LOOP * 3 + 1


def test_recursion_limit_stops_a_runaway_loop(monkeypatch):
    from langgraph.errors import GraphRecursionError

    # Simulate a routing bug that never finishes.
    monkeypatch.setattr(graph, "route_after_critique", lambda state: "rewrite_query")
    deps, _ = make_deps(FakeArxiv({}), FakePaperStore(), FakeCritic([set()]))

    with pytest.raises(GraphRecursionError):
        graph.run_graph(QUESTION, deps=deps)


def test_build_graph_has_the_expected_nodes():
    deps, _ = make_deps(FakeArxiv({}), FakePaperStore(), FakeCritic([set()]))

    nodes = set(build_graph(deps).get_graph().nodes)

    assert {
        "rewrite_query",
        "fetch_candidates",
        "index_candidates",
        "retrieve",
        "critique",
        "finalize",
    } <= nodes


def test_empty_question_is_rejected():
    deps, _ = make_deps(FakeArxiv({}), FakePaperStore(), FakeCritic([set()]))

    with pytest.raises(ValueError):
        graph.run_graph("  ", deps=deps)


def test_history_records_each_iteration():
    arxiv = FakeArxiv(
        {
            "query 1": [paper(n) for n in range(1, 4)],
            "query 2": [paper(n) for n in range(4, 11)],
        }
    )
    critic = FakeCritic([ids(1, 2), ids(1, 2, 5, 6, 7, 8)])
    deps, _ = make_deps(arxiv, FakePaperStore(), critic)

    state = graph.run_graph(QUESTION, deps=deps)

    assert [
        (h.iteration, h.queries, h.relevant, h.candidates) for h in state["history"]
    ] == [
        (1, ["query 1"], 2, 3),
        (2, ["query 2"], 6, 10),
    ]


def test_default_deps_without_fetch_never_calls_arxiv(monkeypatch):
    from app import embeddings, retrieval, sources

    embedder = FakeEmbedder()
    opened = {}

    def fake_get_index(e, *, create=True):
        opened["create"] = create
        return object()

    def fail_fetch(query, max_results):
        raise AssertionError("arXiv must not be called")

    monkeypatch.setattr(embeddings, "get_embedder", lambda: embedder)
    monkeypatch.setattr(retrieval, "get_index", fake_get_index)
    monkeypatch.setattr(sources, "fetch_arxiv", fail_fetch)

    deps = graph.default_deps(fetch=False)

    assert deps.fetch_arxiv("anything", 20) == []
    assert deps.embedder is embedder
    assert opened["create"] is False  # don't create an empty index


def test_mermaid_diagram_shows_nodes_and_conditional_edges():
    diagram = graph.mermaid_diagram()

    for node in ("rewrite_query", "fetch_candidates", "index_candidates", "retrieve"):
        assert node in diagram
    assert "critique -.-> finalize" in diagram
    assert "critique -.-> rewrite_query" in diagram
