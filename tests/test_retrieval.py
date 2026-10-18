"""Tests for app.retrieval using a fake Pinecone index (no network)."""

from __future__ import annotations

from datetime import date

import pytest
from pinecone.models.documents.document import Document

from app import retrieval
from app.retrieval import (
    NAMESPACE,
    PAPER_FIELDS,
    RetrievalConfigError,
    build_filter,
    get_index,
    index_name,
    search,
    upsert_papers,
)
from tests.fakes import FakeEmbedder, FakeIndex, make_paper

# --- build_filter ---------------------------------------------------------


def test_build_filter_without_arguments_is_none():
    assert build_filter() is None


def test_build_filter_year_only():
    assert build_filter(year_min=2022) == {"year": {"$gte": 2022}}


def test_build_filter_categories_only():
    assert build_filter(categories=["cs.LG", "stat.ML"]) == {
        "primary_category": {"$in": ["cs.LG", "stat.ML"]}
    }


def test_build_filter_year_and_categories():
    assert build_filter(year_min=2022, categories=["cs.CL"]) == {
        "year": {"$gte": 2022},
        "primary_category": {"$in": ["cs.CL"]},
    }


@pytest.mark.parametrize("categories", [[], ["", "  "]])
def test_build_filter_ignores_empty_categories(categories):
    assert build_filter(categories=categories) is None


def test_build_filter_cleans_and_dedupes_categories():
    assert build_filter(categories=[" cs.LG", "cs.LG", "cs.AI "]) == {
        "primary_category": {"$in": ["cs.LG", "cs.AI"]}
    }


def test_build_filter_year_zero_is_kept():
    assert build_filter(year_min=0) == {"year": {"$gte": 0}}


# --- search -----------------------------------------------------------------


def stored_match(paper, score: float) -> Document:
    """A match as Pinecone returns it: numbers come back as floats."""
    fields = paper.model_dump(mode="json")
    fields["year"] = float(fields["year"])
    return Document({"_id": paper.arxiv_id, "_score": score, **fields})


def test_search_sends_filter_and_query_to_index():
    embedder = FakeEmbedder()
    index = FakeIndex()

    search(
        "graph networks",
        k=3,
        year_min=2022,
        categories=["cs.LG"],
        embedder=embedder,
        index=index,
    )

    [call] = index.documents.searches
    assert call["namespace"] == NAMESPACE
    assert call["top_k"] == 3
    assert call["filter"] == {
        "year": {"$gte": 2022},
        "primary_category": {"$in": ["cs.LG"]},
    }
    assert call["include_fields"] == PAPER_FIELDS
    [query] = call["score_by"]
    assert query.field == "embedding"
    assert query.values == [14.0, 0.0, 1.0]
    assert embedder.calls == ["graph networks"]


def test_search_without_filters_sends_none():
    index = FakeIndex()

    search("graph networks", embedder=FakeEmbedder(), index=index)

    [call] = index.documents.searches
    assert call["filter"] is None
    assert call["top_k"] == 5


def test_search_returns_papers_with_scores_in_order():
    first = make_paper("2401.00001", title="First")
    second = make_paper(
        "hep-th/9901001", title="Second", year=1999, published=date(1999, 1, 4)
    )
    index = FakeIndex([stored_match(first, 0.91), stored_match(second, 0.42)])

    results = search("anything", embedder=FakeEmbedder(), index=index)

    assert results == [(first, 0.91), (second, 0.42)]
    assert isinstance(results[0][0].year, int)


@pytest.mark.parametrize(("question", "k"), [("", 5), ("   ", 5), ("q", 0)])
def test_search_rejects_invalid_arguments(question, k):
    index = FakeIndex()

    with pytest.raises(ValueError):
        search(question, k=k, embedder=FakeEmbedder(), index=index)

    assert index.documents.searches == []


# --- upsert_papers ----------------------------------------------------------


def test_upsert_papers_batches_of_100():
    papers = [make_paper(f"2401.{i:05d}") for i in range(250)]
    index = FakeIndex()

    count = upsert_papers(papers, FakeEmbedder(), index=index)

    assert count == 250
    sizes = [len(call["documents"]) for call in index.documents.upserts]
    assert sizes == [100, 100, 50]
    assert all(call["namespace"] == NAMESPACE for call in index.documents.upserts)


def test_upsert_papers_document_shape():
    paper = make_paper("2401.01234", title="Attention", abstract="Abstract.")
    index = FakeIndex()

    upsert_papers([paper], FakeEmbedder(), index=index)

    [document] = index.documents.upserts[0]["documents"]
    assert document == {
        "_id": "2401.01234",
        "embedding": [float(len("Attention\n\nAbstract.")), 0.0, 1.0],
        "arxiv_id": "2401.01234",
        "title": "Attention",
        "abstract": "Abstract.",
        "authors": ["Ada Lovelace", "Alan Turing"],
        "published": "2024-01-03",
        "year": 2024,
        "primary_category": "cs.LG",
        "url": "https://arxiv.org/abs/2401.01234",
    }
    assert set(PAPER_FIELDS) <= set(document)


def test_upsert_papers_drops_duplicate_ids_keeping_first():
    papers = [
        make_paper("2401.01234", title="First"),
        make_paper("2401.01234", title="Second"),
    ]
    index = FakeIndex()

    assert upsert_papers(papers, FakeEmbedder(), index=index) == 1
    [document] = index.documents.upserts[0]["documents"]
    assert document["title"] == "First"


def test_upsert_papers_empty_input_does_nothing():
    embedder = FakeEmbedder()
    index = FakeIndex()

    assert upsert_papers([], embedder, index=index) == 0
    assert index.documents.upserts == []
    assert embedder.calls == []


# --- index naming and creation ----------------------------------------------


def test_index_name_includes_embedder_and_dimension():
    assert index_name(FakeEmbedder()) == "papers-fake-3"


def test_index_name_rejects_invalid_names():
    embedder = FakeEmbedder()
    embedder.name = "Bad_Name"

    with pytest.raises(ValueError, match="not a valid Pinecone index name"):
        index_name(embedder)


class FakeIndexes:
    def __init__(self, existing: set[str]) -> None:
        self.existing = existing
        self.created: list[dict] = []

    def exists(self, name: str) -> bool:
        return name in self.existing

    def create(self, **kwargs) -> None:
        self.created.append(kwargs)
        self.existing.add(kwargs["name"])


class FakePinecone:
    def __init__(self, existing: set[str] | None = None) -> None:
        self.indexes = FakeIndexes(existing or set())
        self.opened: list[str] = []

    def index(self, name: str) -> FakeIndex:
        self.opened.append(name)
        return FakeIndex()


def test_get_index_creates_missing_index(monkeypatch):
    pc = FakePinecone()
    monkeypatch.setattr(retrieval, "_pinecone", lambda: pc)

    get_index(FakeEmbedder())

    [created] = pc.indexes.created
    assert created["name"] == "papers-fake-3"
    fields = created["schema"]["fields"]
    assert fields["embedding"]["dimension"] == 3
    assert fields["embedding"]["metric"] == "cosine"
    assert fields["year"]["filterable"] is True
    assert fields["primary_category"]["filterable"] is True
    assert created["deployment"]["cloud"] == "aws"
    assert pc.opened == ["papers-fake-3"]


def test_get_index_reuses_existing_index(monkeypatch):
    pc = FakePinecone(existing={"papers-fake-3"})
    monkeypatch.setattr(retrieval, "_pinecone", lambda: pc)

    get_index(FakeEmbedder())

    assert pc.indexes.created == []
    assert pc.opened == ["papers-fake-3"]


def test_missing_pinecone_key_raises(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    retrieval._pinecone.cache_clear()

    with pytest.raises(RetrievalConfigError, match="PINECONE_API_KEY"):
        retrieval._pinecone()
