"""Test doubles shared across test modules."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

from app.embeddings import Embedder
from app.schemas import Critique, Paper, PaperVerdict, SearchPlan
from app.sources import ArxivFetchError


class FakeEmbedder:
    """A deterministic Embedder: each vector encodes the text's length."""

    name = "fake"
    dimension = 3

    def __init__(self) -> None:
        self.calls: list[object] = []

    def format_paper(self, paper: Paper) -> str:
        return f"{paper.title}\n\n{paper.abstract}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        return [float(len(text)), 0.0, 1.0]


class FakeTokenizer:
    sep_token = "[SEP]"


class FakeSentenceTransformer:
    """Stands in for a SentenceTransformer: no download, records encode calls.

    Each vector is [len(text), 0, 0, ...] with `dimension` entries.
    """

    def __init__(self, dimension: int = 768) -> None:
        self.dimension = dimension
        self.tokenizer = FakeTokenizer()
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        self.calls.append((list(texts), kwargs))
        vectors = np.zeros((len(texts), self.dimension))
        vectors[:, 0] = [len(text) for text in texts]
        return vectors


@dataclass
class FakeMatch:
    """A query match as Pinecone returns it: id, score and stored metadata."""

    id: str
    score: float
    metadata: dict[str, Any]


class FakeQueryResponse:
    def __init__(self, matches: list[FakeMatch]) -> None:
        self.matches = matches


@dataclass
class FakeListItem:
    id: str


@dataclass
class FakeListPage:
    """One page of Index.list: just the IDs."""

    vectors: list[FakeListItem]


@dataclass
class FakeVector:
    id: str
    metadata: dict[str, Any]


@dataclass
class FakeFetchResponse:
    vectors: dict[str, FakeVector]


class FakeIndex:
    """Stands in for a Pinecone Index: records calls, returns canned data.

    `matches` are returned by query; `records` (id -> metadata) are what
    list and fetch see as stored.
    """

    def __init__(
        self,
        matches: list[FakeMatch] | None = None,
        records: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._matches = matches or []
        self.records = records or {}
        self.upserts: list[dict[str, Any]] = []
        self.queries: list[dict[str, Any]] = []
        self.lists: list[dict[str, Any]] = []
        self.fetches: list[dict[str, Any]] = []

    def upsert(self, **kwargs: Any) -> None:
        self.upserts.append(kwargs)

    def query(self, **kwargs: Any) -> FakeQueryResponse:
        self.queries.append(kwargs)
        return FakeQueryResponse(self._matches)

    def list(self, **kwargs: Any) -> Iterator[FakeListPage]:
        """Like Pinecone: IDs in sorted order, `limit` per page, no empty pages."""
        self.lists.append(kwargs)
        ids = sorted(self.records)
        limit = kwargs.get("limit") or 100
        for start in range(0, len(ids), limit):
            yield FakeListPage([FakeListItem(i) for i in ids[start : start + limit]])

    def fetch(self, **kwargs: Any) -> FakeFetchResponse:
        """Like Pinecone: unknown IDs are silently left out."""
        self.fetches.append(kwargs)
        return FakeFetchResponse(
            {
                i: FakeVector(i, self.records[i])
                for i in kwargs["ids"]
                if i in self.records
            }
        )


def make_paper(arxiv_id: str = "2401.01234", **overrides: Any) -> Paper:
    fields: dict[str, Any] = {
        "arxiv_id": arxiv_id,
        "title": "A Title",
        "abstract": "An abstract.",
        "authors": ["Ada Lovelace", "Alan Turing"],
        "published": date(2024, 1, 3),
        "year": 2024,
        "primary_category": "cs.LG",
        "url": f"https://arxiv.org/abs/{arxiv_id}",
    }
    fields.update(overrides)
    return Paper(**fields)


class FakeChatModel:
    """Stands in for a chat model: no network, returns one canned structured reply.

    with_structured_output records the requested schema and returns self, so
    invoke records the messages and returns `response` (a Pydantic object).
    """

    def __init__(self, response: Any) -> None:
        self.response = response
        self.schemas: list[type] = []
        self.calls: list[list[tuple[str, str]]] = []

    def with_structured_output(self, schema: type) -> FakeChatModel:
        self.schemas.append(schema)
        return self

    def invoke(self, messages: list[tuple[str, str]]) -> Any:
        self.calls.append(messages)
        return self.response


# Fakes for the agent graph's dependencies (see app.graph.GraphDeps).


class FakeRewriter:
    """Returns ["query 1"] on the first call, ["query 2"] on the second, and so on."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def __call__(self, question: str, feedback: str | None = None) -> SearchPlan:
        self.calls.append((question, feedback))
        return SearchPlan(queries=[f"query {len(self.calls)}"])


class FakeArxiv:
    """Returns canned papers per query; raises ArxivFetchError if `fail` is set."""

    def __init__(self, by_query: dict[str, list[Paper]], fail: bool = False) -> None:
        self.by_query = by_query
        self.fail = fail
        self.calls: list[tuple[str, int]] = []

    def __call__(self, query: str, max_results: int = 50) -> list[Paper]:
        self.calls.append((query, max_results))
        if self.fail:
            raise ArxivFetchError(f"arXiv search failed for query {query!r}: 503")
        return self.by_query.get(query, [])


class FakePaperStore:
    """An in-memory vector index: upsert stores papers, search ranks them by score."""

    def __init__(self, papers: Sequence[Paper] = ()) -> None:
        self.papers = {p.arxiv_id: p for p in papers}
        self.upserts: list[list[str]] = []
        self.searches: list[tuple[str, int]] = []

    def upsert(self, papers: Sequence[Paper], embedder: Embedder) -> int:
        self.upserts.append([p.arxiv_id for p in papers])
        self.papers.update({p.arxiv_id: p for p in papers})
        return len(papers)

    def search(
        self, question: str, k: int = 5, *, embedder: Embedder
    ) -> list[tuple[Paper, float]]:
        self.searches.append((question, k))
        scored = [(p, int(p.arxiv_id[-5:]) / 100) for p in self.papers.values()]
        return sorted(scored, key=lambda pair: pair[1], reverse=True)[:k]


class FakeCritic:
    """Call i marks the ids in relevant_sets[i] relevant (the last set repeats)."""

    def __init__(self, relevant_sets: list[set[str]]) -> None:
        self.relevant_sets = relevant_sets
        self.calls: list[list[str]] = []

    def __call__(self, question: str, papers: Sequence[Paper]) -> Critique:
        i = len(self.calls)
        self.calls.append([p.arxiv_id for p in papers])
        relevant = self.relevant_sets[min(i, len(self.relevant_sets) - 1)]
        return Critique(
            verdicts=[
                PaperVerdict(
                    arxiv_id=p.arxiv_id,
                    relevant=p.arxiv_id in relevant,
                    reason=f"reason {p.arxiv_id}",
                )
                for p in papers
            ],
            feedback=f"feedback {i}",
        )
