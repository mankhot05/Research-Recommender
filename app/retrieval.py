"""Vector search over the Pinecone index of paper abstracts.

Each embedder gets its own index (papers-<name>-<dimension>), so vectors from
different models never share an index. Papers are stored as documents keyed by
arxiv_id, so re-ingesting a paper overwrites it instead of duplicating it.
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
VECTOR_FIELD = "embedding"
UPSERT_BATCH_SIZE = 100
CLOUD = "aws"
REGION = "us-east-1"

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
    """Return the embedder's index, creating it (serverless, cosine) if missing."""
    from pinecone import SchemaBuilder

    pc = _pinecone()
    name = index_name(embedder)
    if not pc.indexes.exists(name):
        # Numbers are stored as floats, so year is a filterable float field.
        schema = (
            SchemaBuilder()
            .add_dense_vector_field(
                VECTOR_FIELD, dimension=embedder.dimension, metric="cosine"
            )
            .add_float_field("year", filterable=True)
            .add_string_field("primary_category", filterable=True)
            .build()
        )
        # Blocks until the index is ready to accept writes.
        pc.indexes.create(
            name=name,
            schema=schema,
            deployment={
                "deployment_type": "managed",
                "cloud": CLOUD,
                "region": REGION,
            },
        )
    return pc.index(name)


def paper_to_document(paper: Paper, vector: Sequence[float]) -> dict[str, Any]:
    """Every Paper field plus the vector, keyed by arxiv_id."""
    return {
        "_id": paper.arxiv_id,
        VECTOR_FIELD: list(vector),
        **paper.model_dump(mode="json"),
    }


def document_to_paper(fields: dict[str, Any]) -> Paper:
    """Rebuild a Paper from stored fields; Paper's validators re-check them."""
    return Paper.model_validate(fields)


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
        documents = [
            paper_to_document(p, v) for p, v in zip(batch, vectors, strict=True)
        ]
        index.documents.upsert(namespace=NAMESPACE, documents=documents)
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
    from pinecone import DenseVectorQuery

    if not question.strip():
        raise ValueError("question must not be empty")
    if k < 1:
        raise ValueError("k must be at least 1")
    if embedder is None:
        embedder = get_embedder()
    if index is None:
        index = get_index(embedder)

    response = index.documents.search(
        namespace=NAMESPACE,
        top_k=k,
        score_by=[
            DenseVectorQuery(field=VECTOR_FIELD, values=embedder.embed_query(question))
        ],
        include_fields=PAPER_FIELDS,
        filter=build_filter(year_min, categories),
    )
    return [
        (
            document_to_paper({f: match.get(f) for f in PAPER_FIELDS}),
            float(match.score),
        )
        for match in response.matches
    ]
