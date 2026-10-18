"""Test doubles shared across test modules."""

from __future__ import annotations

from datetime import date
from typing import Any

from pinecone.models.documents.document import Document

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


class FakeSearchResponse:
    def __init__(self, matches: list[Document]) -> None:
        self.matches = matches


class FakeDocuments:
    """Stands in for Index.documents: records calls, returns canned matches."""

    def __init__(self, matches: list[Document]) -> None:
        self._matches = matches
        self.upserts: list[dict[str, Any]] = []
        self.searches: list[dict[str, Any]] = []

    def upsert(self, **kwargs: Any) -> None:
        self.upserts.append(kwargs)

    def search(self, **kwargs: Any) -> FakeSearchResponse:
        self.searches.append(kwargs)
        return FakeSearchResponse(self._matches)


class FakeIndex:
    def __init__(self, matches: list[Document] | None = None) -> None:
        self.documents = FakeDocuments(matches or [])


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
