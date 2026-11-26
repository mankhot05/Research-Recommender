"""Tests for app.retrieval using a fake Pinecone index (no network)."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest
from pinecone import ServerlessSpec
from pinecone.models.indexes.schema import (
    DenseVectorField,
    FloatField,
    IndexSchema,
    StringField,
)

from app import retrieval
from app.retrieval import (
    NAMESPACE,
    PAPER_FIELDS,
    RetrievalConfigError,
    build_filter,
    export_papers,
    get_index,
    index_name,
    open_index,
    paper_to_record,
    search,
    upsert_papers,
)
from tests.fakes import FakeEmbedder, FakeIndex, FakeMatch, make_paper

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


def stored_match(paper, score: float) -> FakeMatch:
    """A match as Pinecone returns it: numbers in metadata come back as floats."""
    metadata = paper.model_dump(mode="json")
    metadata["year"] = float(metadata["year"])
    return FakeMatch(id=paper.arxiv_id, score=score, metadata=metadata)


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

    [call] = index.queries
    assert call == {
        "vector": [14.0, 0.0, 1.0],
        "top_k": 3,
        "namespace": NAMESPACE,
        "filter": {
            "year": {"$gte": 2022},
            "primary_category": {"$in": ["cs.LG"]},
        },
        "include_metadata": True,
    }
    assert embedder.calls == ["graph networks"]


def test_search_without_filters_sends_none():
    index = FakeIndex()

    search("graph networks", embedder=FakeEmbedder(), index=index)

    [call] = index.queries
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

    assert index.queries == []


# --- upsert_papers ----------------------------------------------------------


def test_upsert_papers_batches_of_100():
    papers = [make_paper(f"2401.{i:05d}") for i in range(250)]
    index = FakeIndex()

    count = upsert_papers(papers, FakeEmbedder(), index=index)

    assert count == 250
    sizes = [len(call["vectors"]) for call in index.upserts]
    assert sizes == [100, 100, 50]
    assert all(call["namespace"] == NAMESPACE for call in index.upserts)


def test_upsert_papers_record_shape():
    paper = make_paper("2401.01234", title="Attention", abstract="Abstract.")
    index = FakeIndex()

    upsert_papers([paper], FakeEmbedder(), index=index)

    [record] = index.upserts[0]["vectors"]
    assert record == {
        "id": "2401.01234",
        "values": [float(len("Attention\n\nAbstract.")), 0.0, 1.0],
        "metadata": {
            "arxiv_id": "2401.01234",
            "title": "Attention",
            "abstract": "Abstract.",
            "authors": ["Ada Lovelace", "Alan Turing"],
            "published": "2024-01-03",
            "year": 2024,
            "primary_category": "cs.LG",
            "url": "https://arxiv.org/abs/2401.01234",
        },
    }
    assert set(record["metadata"]) == set(PAPER_FIELDS)


def test_upsert_papers_embeds_the_embedders_own_format():
    class SepEmbedder(FakeEmbedder):
        def format_paper(self, paper):
            return f"{paper.title}[SEP]{paper.abstract}"

    embedder = SepEmbedder()

    upsert_papers([make_paper(title="T", abstract="A")], embedder, index=FakeIndex())

    assert embedder.calls == [["T[SEP]A"]]


def test_upsert_papers_drops_duplicate_ids_keeping_first():
    papers = [
        make_paper("2401.01234", title="First"),
        make_paper("2401.01234", title="Second"),
    ]
    index = FakeIndex()

    assert upsert_papers(papers, FakeEmbedder(), index=index) == 1
    [record] = index.upserts[0]["vectors"]
    assert record["metadata"]["title"] == "First"


def test_upsert_papers_empty_input_does_nothing():
    embedder = FakeEmbedder()
    index = FakeIndex()

    assert upsert_papers([], embedder, index=index) == 0
    assert index.upserts == []
    assert embedder.calls == []


# --- index naming and creation ----------------------------------------------


def test_index_name_includes_embedder_and_dimension():
    assert index_name(FakeEmbedder()) == "papers-fake-3"


def test_index_name_rejects_invalid_names():
    embedder = FakeEmbedder()
    embedder.name = "Bad_Name"

    with pytest.raises(ValueError, match="not a valid Pinecone index name"):
        index_name(embedder)


PLAIN_SCHEMA = IndexSchema(
    fields={"_values": DenseVectorField(dimension=3, metric="cosine")}
)
OLD_SCHEMA = IndexSchema(
    fields={
        "embedding": DenseVectorField(dimension=3, metric="cosine"),
        "year": FloatField(filterable=True),
        "primary_category": StringField(filterable=True),
    }
)


class FakeIndexes:
    def __init__(self, existing: dict[str, IndexSchema]) -> None:
        self.existing = existing
        self.created: list[dict] = []
        self.deleted: list[str] = []

    def exists(self, name: str) -> bool:
        return name in self.existing

    def describe(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(name=name, schema=self.existing[name])

    def create(self, **kwargs) -> None:
        self.created.append(kwargs)
        self.existing[kwargs["name"]] = PLAIN_SCHEMA

    def delete(self, name: str) -> None:
        self.deleted.append(name)


class FakePinecone:
    def __init__(self, existing: dict[str, IndexSchema] | None = None) -> None:
        self.indexes = FakeIndexes(existing or {})
        self.opened: list[str] = []

    def index(self, name: str) -> FakeIndex:
        self.opened.append(name)
        return FakeIndex()


def test_get_index_creates_plain_serverless_index(monkeypatch):
    pc = FakePinecone()
    monkeypatch.setattr(retrieval, "_pinecone", lambda: pc)

    get_index(FakeEmbedder())

    [created] = pc.indexes.created
    assert created == {
        "name": "papers-fake-3",
        "dimension": 3,
        "metric": "cosine",
        "spec": ServerlessSpec(cloud="aws", region="us-east-1"),
    }
    assert "schema" not in created
    assert pc.opened == ["papers-fake-3"]


def test_get_index_reuses_existing_plain_index(monkeypatch):
    pc = FakePinecone(existing={"papers-fake-3": PLAIN_SCHEMA})
    monkeypatch.setattr(retrieval, "_pinecone", lambda: pc)

    get_index(FakeEmbedder())

    assert pc.indexes.created == []
    assert pc.opened == ["papers-fake-3"]


def test_get_index_without_create_refuses_missing_index(monkeypatch):
    pc = FakePinecone()
    monkeypatch.setattr(retrieval, "_pinecone", lambda: pc)

    with pytest.raises(RetrievalConfigError, match="'papers-fake-3' does not exist"):
        get_index(FakeEmbedder(), create=False)

    assert pc.indexes.created == []
    assert pc.opened == []


def test_get_index_without_create_returns_existing_index(monkeypatch):
    pc = FakePinecone(existing={"papers-fake-3": PLAIN_SCHEMA})
    monkeypatch.setattr(retrieval, "_pinecone", lambda: pc)

    get_index(FakeEmbedder(), create=False)

    assert pc.opened == ["papers-fake-3"]


def test_get_index_refuses_index_with_metadata_schema(monkeypatch):
    pc = FakePinecone(existing={"papers-fake-3": OLD_SCHEMA})
    monkeypatch.setattr(retrieval, "_pinecone", lambda: pc)

    with pytest.raises(RetrievalConfigError) as error:
        get_index(FakeEmbedder())

    message = str(error.value)
    assert "papers-fake-3" in message
    assert "primary_category" in message and "year" in message
    assert pc.indexes.created == []
    assert pc.indexes.deleted == []
    assert pc.opened == []


def test_missing_pinecone_key_raises(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    retrieval._pinecone.cache_clear()

    with pytest.raises(RetrievalConfigError, match="PINECONE_API_KEY"):
        retrieval._pinecone()


def test_open_index_opens_existing_plain_index(monkeypatch):
    pc = FakePinecone(existing={"papers-openai-1536": PLAIN_SCHEMA})
    monkeypatch.setattr(retrieval, "_pinecone", lambda: pc)

    open_index("papers-openai-1536")

    assert pc.opened == ["papers-openai-1536"]
    assert pc.indexes.created == []


def test_open_index_refuses_missing_index(monkeypatch):
    pc = FakePinecone()
    monkeypatch.setattr(retrieval, "_pinecone", lambda: pc)

    with pytest.raises(RetrievalConfigError, match="'nope' does not exist"):
        open_index("nope")

    assert pc.indexes.created == []
    assert pc.opened == []


# --- export_papers --------------------------------------------------------


def stored(*papers):
    """Index records (id -> metadata) exactly as upsert_papers stores them."""
    return {p.arxiv_id: paper_to_record(p, [0.0, 0.0, 1.0])["metadata"] for p in papers}


def test_export_papers_round_trips_stored_metadata():
    papers = [make_paper("2401.00001", title="One"), make_paper("2401.00002")]
    index = FakeIndex(records=stored(*papers))

    assert export_papers(index) == papers


def test_export_papers_fetches_one_batch_per_listed_page():
    papers = [make_paper(f"2401.{i:05d}") for i in range(250)]
    index = FakeIndex(records=stored(*papers))

    exported = export_papers(index, batch_size=100)

    assert exported == papers
    assert index.lists == [{"namespace": NAMESPACE, "limit": 100}]
    assert [len(f["ids"]) for f in index.fetches] == [100, 100, 50]
    assert all(f["namespace"] == NAMESPACE for f in index.fetches)


def test_export_papers_empty_index():
    index = FakeIndex()

    assert export_papers(index) == []
    assert index.fetches == []


def test_export_papers_skips_ids_missing_from_fetch():
    papers = [make_paper("2401.00001"), make_paper("2401.00002")]
    index = FakeIndex(records=stored(*papers))
    real_fetch = index.fetch

    def fetch_without_second(**kwargs):
        response = real_fetch(**kwargs)
        response.vectors.pop("2401.00002", None)  # deleted after it was listed
        return response

    index.fetch = fetch_without_second

    assert export_papers(index) == [papers[0]]


def test_export_papers_drops_duplicate_arxiv_ids():
    paper = make_paper("2401.00001")
    # Two vector IDs whose metadata name the same paper.
    index = FakeIndex(
        records={"a": stored(paper)[paper.arxiv_id], "b": stored(paper)[paper.arxiv_id]}
    )

    assert export_papers(index) == [paper]


def test_export_papers_rejects_invalid_metadata():
    index = FakeIndex(records={"2401.00001": {"title": "no other fields"}})

    with pytest.raises(ValueError, match="'2401.00001'"):
        export_papers(index)
