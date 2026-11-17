"""Vector search over the Pinecone index of paper abstracts.

Each embedder gets its own index (papers-<name>-<dimension>), so vectors from
different models never share an index. The index is a plain serverless dense
index: it declares only the vector, and every Paper field is stored as per-record
metadata (Pinecone indexes metadata automatically, so year/category filters work).
Records are keyed by arxiv_id, so re-ingesting a paper overwrites it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from app.config import get_settings
from app.embeddings import Embedder, get_embedder, paper_to_text
from app.schemas import Paper

if TYPE_CHECKING:
    from pinecone import Pinecone
    from pinecone.index import Index

NAMESPACE = "arxiv"
UPSERT_BATCH_SIZE = 100
CLOUD = "aws"
REGION = "us-east-1"
METRIC = "cosine"

PAPER_FIELDS = list(Paper.model_fields)

# Pinecone index names: lowercase letters, digits and hyphens, at most 45 chars.
_INDEX_NAME_RE = re.compile(r"^[a-z0-9-]{1,45}$")


class RetrievalConfigError(RuntimeError):
    """Pinecone cannot be reached with the current configuration."""


def index_name(embedder: Embedder) -> str:
    """The index for this embedder, e.g. papers-openai-1536."""
    name = f"papers-{embedder.name}-{embedder.dimension}"
    if not _INDEX_NAME_RE.match(name):
        raise ValueError(f"{name!r} is not a valid Pinecone index name")
    return name


@lru_cache(maxsize=1)
def _pinecone() -> Pinecone:
    from pinecone import Pinecone

    api_key = get_settings().pinecone_api_key
    if api_key is None or not api_key.get_secret_value().strip():
        raise RetrievalConfigError("Missing PINECONE_API_KEY. Set it in .env")
    return Pinecone(api_key=api_key.get_secret_value().strip())


def get_index(embedder: Embedder) -> Index:
    """Return the embedder's index, creating it (serverless, cosine) if missing.

    Raises RetrievalConfigError if an existing index declares metadata fields in
    its schema; that index must be deleted by hand before ingesting again.
    """
    from pinecone import ServerlessSpec

    pc = _pinecone()
    name = index_name(embedder)
    if pc.indexes.exists(name):
        _check_plain_dense_index(pc, name)
    else:
        # Blocks until the index is ready to accept writes.
        pc.indexes.create(
            name=name,
            dimension=embedder.dimension,
            metric=METRIC,
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )
    return pc.index(name)


def _check_plain_dense_index(pc: Pinecone, name: str) -> None:
    """Fail loudly if the index declares anything besides vector fields."""
    from pinecone.models.indexes.schema import DenseVectorField, SparseVectorField

    fields = pc.indexes.describe(name).schema.fields
    extra = sorted(
        field_name
        for field_name, field in fields.items()
        if not isinstance(field, (DenseVectorField, SparseVectorField))
    )
    if extra:
        raise RetrievalConfigError(
            f"Index {name!r} declares metadata fields in its schema ({', '.join(extra)}). "
            "It was created with an old configuration and is not deleted automatically. "
            "Delete it in the Pinecone console (or with pc.indexes.delete) and re-run."
        )


def paper_to_record(paper: Paper, vector: Sequence[float]) -> dict[str, Any]:
    """A Pinecone vector record: id, values, and every Paper field as metadata."""
    return {
        "id": paper.arxiv_id,
        "values": list(vector),
        "metadata": paper.model_dump(mode="json"),
    }


def metadata_to_paper(metadata: dict[str, Any]) -> Paper:
    """Rebuild a Paper from stored metadata; Paper's validators re-check it."""
    return Paper.model_validate(metadata)


def upsert_papers(
    papers: Sequence[Paper],
    embedder: Embedder,
    *,
    index: Index | None = None,
) -> int:
    """Embed and upsert papers in batches of 100. Returns how many were upserted."""
    # Pinecone rejects a batch with duplicate _ids; keep the first of each.
    by_id: dict[str, Paper] = {}
    for paper in papers:
        by_id.setdefault(paper.arxiv_id, paper)
    unique = list(by_id.values())
    if not unique:
        return 0
    if index is None:
        index = get_index(embedder)

    for start in range(0, len(unique), UPSERT_BATCH_SIZE):
        batch = unique[start : start + UPSERT_BATCH_SIZE]
        vectors = embedder.embed_documents([paper_to_text(p) for p in batch])
        records = [paper_to_record(p, v) for p, v in zip(batch, vectors, strict=True)]
        index.upsert(vectors=records, namespace=NAMESPACE, show_progress=False)
    return len(unique)


def build_filter(
    year_min: int | None = None,
    categories: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    """A Pinecone metadata filter; multiple top-level keys are ANDed together."""
    conditions: dict[str, Any] = {}
    if year_min is not None:
        conditions["year"] = {"$gte": year_min}
    cleaned = list(dict.fromkeys(c.strip() for c in categories or () if c.strip()))
    if cleaned:
        conditions["primary_category"] = {"$in": cleaned}
    return conditions or None


def search(
    question: str,
    k: int = 5,
    year_min: int | None = None,
    categories: Sequence[str] | None = None,
    *,
    embedder: Embedder | None = None,
    index: Index | None = None,
) -> list[tuple[Paper, float]]:
    """The k papers most similar to the question, best first, with their scores."""
    if not question.strip():
        raise ValueError("question must not be empty")
    if k < 1:
        raise ValueError("k must be at least 1")
    if embedder is None:
        embedder = get_embedder()
    if index is None:
        index = get_index(embedder)

    response = index.query(
        vector=embedder.embed_query(question),
        top_k=k,
        namespace=NAMESPACE,
        filter=build_filter(year_min, categories),
        include_metadata=True,
    )
    return [
        (metadata_to_paper(dict(match.metadata or {})), float(match.score))
        for match in response.matches
    ]
