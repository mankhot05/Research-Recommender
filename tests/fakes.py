"""Test doubles shared across test modules."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from app.schemas import Paper


class FakeEmbedder:
    """A deterministic Embedder: each vector encodes the text's length."""

    name = "fake"
    dimension = 3

    def __init__(self) -> None:
        self.calls: list[object] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        return [float(len(text)), 0.0, 1.0]


@dataclass
class FakeMatch:
    """A query match as Pinecone returns it: id, score and stored metadata."""

    id: str
    score: float
    metadata: dict[str, Any]


class FakeQueryResponse:
    def __init__(self, matches: list[FakeMatch]) -> None:
        self.matches = matches


class FakeIndex:
    """Stands in for a Pinecone Index: records calls, returns canned matches."""

    def __init__(self, matches: list[FakeMatch] | None = None) -> None:
        self._matches = matches or []
        self.upserts: list[dict[str, Any]] = []
        self.queries: list[dict[str, Any]] = []

    def upsert(self, **kwargs: Any) -> None:
        self.upserts.append(kwargs)

    def query(self, **kwargs: Any) -> FakeQueryResponse:
        self.queries.append(kwargs)
        return FakeQueryResponse(self._matches)


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
