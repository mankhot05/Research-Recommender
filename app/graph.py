"""LangGraph agent loop: rewrite -> search -> retrieve -> critique -> loop or finish.

    START -> rewrite_query -> fetch_candidates -> index_candidates -> retrieve
          -> critique --(enough relevant, or out of iterations)--> finalize -> END
                      \\--(otherwise, with feedback)--> rewrite_query

Every node reads the shared AgentState and returns only the keys it changes;
LangGraph merges that partial update into the state before the next node runs.
Dependencies (LLM functions, arXiv fetcher, embedder, index) come in through
GraphDeps so tests can swap in fakes.
"""

from __future__ import annotations

import logging
import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Annotated, Any, Literal, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.embeddings import Embedder
from app.schemas import (
    Critique,
    GradedPaper,
    IterationSummary,
    Paper,
    Recommendation,
    SearchPlan,
)
from app.sources import ArxivFetchError

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 3
FETCH_PER_QUERY = 20
RETRIEVE_K = 10
MIN_RELEVANT = 5
TOP_N = 5
NODES_PER_LOOP = 5  # rewrite, fetch, index, retrieve, critique


class AgentState(TypedDict, total=False):
    """Everything the nodes share. total=False: keys appear as nodes fill them in."""

    question: str  # the user's original question; never rewritten
    search_queries: list[str]  # arXiv queries for this iteration
    iteration: int  # how many rewrite passes have run
    max_iterations: int
    fetched: list[Paper]  # new papers from arXiv this iteration
    candidates: list[tuple[Paper, float]]  # top-k from the index, with scores
    graded: list[GradedPaper]  # candidates after critique
    feedback: str | None  # critique's note on what's missing
    final: list[Recommendation]
    # A reducer: operator.add makes LangGraph APPEND each update to this list
    # instead of replacing it, so one summary per iteration survives the loop.
    history: Annotated[list[IterationSummary], operator.add]


@dataclass
class GraphDeps:
    """The outside world the graph talks to. Signatures match the real functions."""

    rewrite_queries: Callable[[str, str | None], SearchPlan]
    critique: Callable[[str, Sequence[Paper]], Critique]
    fetch_arxiv: Callable[[str, int], list[Paper]]
    embedder: Embedder
    upsert_papers: Callable[[Sequence[Paper], Embedder], int]
    search: Callable[..., list[tuple[Paper, float]]]  # (question, k=, embedder=)


def _no_fetch(query: str, max_results: int) -> list[Paper]:
    return []


def default_deps(*, fetch: bool = True) -> GraphDeps:
    """The real dependencies: OpenAI, arXiv, the configured embedder and its index.

    With fetch=False arXiv is never called, so only papers already in the index
    can be recommended; the index must already exist (it is not created).
    """
    from app import llm, retrieval, sources
    from app.embeddings import get_embedder

    embedder = get_embedder()
    # Opened once, shared by upsert and search.
    index = retrieval.get_index(embedder, create=fetch)
    return GraphDeps(
        rewrite_queries=llm.rewrite_queries,
        critique=llm.critique,
        fetch_arxiv=sources.fetch_arxiv if fetch else _no_fetch,
        embedder=embedder,
        upsert_papers=partial(retrieval.upsert_papers, index=index),
        search=partial(retrieval.search, index=index),
    )


def route_after_critique(state: AgentState) -> Literal["finalize", "rewrite_query"]:
    """Finish when there are enough relevant papers or no iterations are left."""
    relevant = sum(1 for g in state.get("graded", []) if g.relevant)
    if relevant >= MIN_RELEVANT or state["iteration"] >= state["max_iterations"]:
        return "finalize"
    return "rewrite_query"


def recursion_limit(max_iterations: int) -> int:
    """LangGraph's step cap: every loop's nodes, plus finalize, plus a margin.

    A safety net: if routing ever loops by mistake, LangGraph raises
    GraphRecursionError instead of calling the LLM forever.
    """
    return NODES_PER_LOOP * max_iterations + 5


def build_graph(deps: GraphDeps) -> CompiledStateGraph:
    """Wire the nodes into a compiled graph. Each node is a closure over deps."""

    def rewrite_query(state: AgentState) -> dict[str, Any]:
        plan = deps.rewrite_queries(state["question"], state.get("feedback"))
        return {"search_queries": plan.queries, "iteration": state["iteration"] + 1}

    def fetch_candidates(state: AgentState) -> dict[str, Any]:
        fetched: list[Paper] = []
        for query in state["search_queries"]:
            try:
                fetched.extend(deps.fetch_arxiv(query, FETCH_PER_QUERY))
            except ArxivFetchError as e:
                logger.warning("%s; continuing with the existing index", e)
        return {"fetched": fetched}

    def index_candidates(state: AgentState) -> dict[str, Any]:
        if state["fetched"]:
            deps.upsert_papers(state["fetched"], deps.embedder)
        return {}

    def retrieve(state: AgentState) -> dict[str, Any]:
        # Search with the ORIGINAL question: rewritten queries are for arXiv's
        # keyword search; the vector index ranks by meaning of the real question.
        candidates = deps.search(
            state["question"], k=RETRIEVE_K, embedder=deps.embedder
        )
        return {"candidates": candidates}

    def critique(state: AgentState) -> dict[str, Any]:
        candidates = state["candidates"]
        result = deps.critique(state["question"], [paper for paper, _ in candidates])
        verdicts = {v.arxiv_id: v for v in result.verdicts}
        graded = []
        for paper, score in candidates:
            verdict = verdicts.get(paper.arxiv_id)
            graded.append(
                GradedPaper(
                    paper=paper,
                    score=score,
                    relevant=verdict is not None and verdict.relevant,
                    reason=verdict.reason if verdict else "Not graded.",
                )
            )
        summary = IterationSummary(
            iteration=state["iteration"],
            queries=state["search_queries"],
            relevant=sum(1 for g in graded if g.relevant),
            candidates=len(graded),
        )
        return {"graded": graded, "feedback": result.feedback, "history": [summary]}

    def finalize(state: AgentState) -> dict[str, Any]:
        relevant = [g for g in state.get("graded", []) if g.relevant]
        relevant.sort(key=lambda g: g.score, reverse=True)
        final = [
            Recommendation(paper=g.paper, score=g.score, reason=g.reason)
            for g in relevant[:TOP_N]
        ]
        return {"final": final}

    graph = StateGraph(AgentState)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("fetch_candidates", fetch_candidates)
    graph.add_node("index_candidates", index_candidates)
    graph.add_node("retrieve", retrieve)
    graph.add_node("critique", critique)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "fetch_candidates")
    graph.add_edge("fetch_candidates", "index_candidates")
    graph.add_edge("index_candidates", "retrieve")
    graph.add_edge("retrieve", "critique")
    graph.add_conditional_edges("critique", route_after_critique)
    graph.add_edge("finalize", END)
    return graph.compile()


def run_graph(
    question: str,
    *,
    deps: GraphDeps | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> AgentState:
    """Run the agent and return the full final state (handy for debugging)."""
    if not question.strip():
        raise ValueError("question must not be empty")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    graph = build_graph(deps or default_deps())
    initial: AgentState = {
        "question": question,
        "iteration": 0,
        "max_iterations": max_iterations,
        "feedback": None,
        "history": [],
    }
    return graph.invoke(
        initial, config={"recursion_limit": recursion_limit(max_iterations)}
    )


def run_agent(
    question: str,
    *,
    deps: GraphDeps | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> list[Recommendation]:
    """Answer a research question with up to TOP_N relevant papers, best first."""
    return run_graph(question, deps=deps, max_iterations=max_iterations)["final"]


def _not_called(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("placeholder dependency: only for drawing the graph")


def mermaid_diagram() -> str:
    """The graph as Mermaid text. Needs no API keys: building never calls deps."""
    placeholder = GraphDeps(
        rewrite_queries=_not_called,
        critique=_not_called,
        fetch_arxiv=_not_called,
        embedder=cast(Embedder, None),
        upsert_papers=_not_called,
        search=_not_called,
    )
    return build_graph(placeholder).get_graph().draw_mermaid()
