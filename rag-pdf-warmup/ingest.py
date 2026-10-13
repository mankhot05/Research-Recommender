"""Ingest every PDF in data/ into the Pinecone index `pdf-warmup`.

Pipeline: load pages -> split into chunks -> embed chunks -> upsert to Pinecone.
Chunk IDs are derived from (filename, page, chunk index), so re-running the
script overwrites existing documents instead of duplicating them.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_text_splitters import RecursiveCharacterTextSplitter

if TYPE_CHECKING:
    from openai import OpenAI
    from pinecone import Pinecone
    from pinecone.index import Index

DATA_DIR = Path(__file__).parent / "data"

INDEX_NAME = "pdf-warmup"
NAMESPACE = "pdfs"
VECTOR_FIELD = "embedding"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

EMBED_BATCH_SIZE = 100
UPSERT_BATCH_SIZE = 100


@dataclass(frozen=True)
class Page:
    """Text of one PDF page. `page` is 1-based, matching what a PDF viewer shows."""

    source: str
    page: int
    text: str


@dataclass(frozen=True)
class Chunk:
    """A piece of a page small enough to embed and retrieve on its own."""

    id: str
    source: str
    page: int
    chunk_index: int
    text: str


def load_pages(data_dir: Path) -> list[Page]:
    """Extract text page-by-page from every PDF in `data_dir`, skipping blank pages."""
    from pypdf import PdfReader

    pdf_paths = sorted(p for p in data_dir.iterdir() if p.suffix.lower() == ".pdf")
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {data_dir}")

    pages: list[Page] = []
    for path in pdf_paths:
        reader = PdfReader(path)
        for page_number, pdf_page in enumerate(reader.pages, start=1):
            text = pdf_page.extract_text() or ""
            if text.strip():
                pages.append(Page(source=path.name, page=page_number, text=text))
    return pages


def _slugify(value: str) -> str:
    """Lowercase ASCII slug: accents folded, every other non-alphanumeric run -> '-'."""
    decomposed = unicodedata.normalize("NFKD", value)
    without_marks = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "-", without_marks.lower()).strip("-")


def make_chunk_id(source: str, page: int, chunk_index: int) -> str:
    """Deterministic, ASCII-only ID (Pinecone rejects non-ASCII IDs).

    The slug keeps IDs readable; the short filename hash keeps them unique when
    two filenames slugify to the same text (e.g. "a b.pdf" and "a-b.pdf").
    """
    stem = _slugify(Path(source).stem) or "doc"
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
    return f"{stem}-{digest}-p{page}-c{chunk_index}"


def chunk_pages(
    pages: Sequence[Page],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split each page into overlapping chunks. Chunk indexes restart at 0 per page."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: list[Chunk] = []
    for page in pages:
        for chunk_index, text in enumerate(splitter.split_text(page.text)):
            chunks.append(
                Chunk(
                    id=make_chunk_id(page.source, page.page, chunk_index),
                    source=page.source,
                    page=page.page,
                    chunk_index=chunk_index,
                    text=text,
                )
            )
    return chunks


def embed_texts(
    client: OpenAI,
    texts: Sequence[str],
    batch_size: int = EMBED_BATCH_SIZE,
) -> list[list[float]]:
    """Embed `texts` in batches, returning vectors in the same order as the input."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        if len(response.data) != len(batch):
            raise RuntimeError(
                f"OpenAI returned {len(response.data)} embeddings for {len(batch)} inputs"
            )
        # Sort by `index` so order never depends on response ordering.
        vectors.extend(
            item.embedding for item in sorted(response.data, key=lambda d: d.index)
        )
    return vectors


def ensure_index(pc: Pinecone) -> Index:
    """Return the `pdf-warmup` index, creating it (serverless, cosine, 1536-d) if missing."""
    if not pc.indexes.exists(INDEX_NAME):
        print(f"Creating index {INDEX_NAME!r}...")
        # Blocks until the index is ready to accept writes.
        pc.indexes.create(
            name=INDEX_NAME,
            schema={
                "fields": {
                    VECTOR_FIELD: {
                        "type": "dense_vector",
                        "dimension": EMBEDDING_DIMENSION,
                        "metric": "cosine",
                    }
                }
            },
            deployment={
                "deployment_type": "managed",
                "cloud": "aws",
                "region": "us-east-1",
            },
        )
    return pc.index(INDEX_NAME)


def upsert_chunks(
    index: Index,
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]],
    batch_size: int = UPSERT_BATCH_SIZE,
) -> None:
    """Write each chunk as a document: its vector plus source/page/text metadata."""
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")

    documents = [
        {
            "_id": chunk.id,
            VECTOR_FIELD: list(vector),
            "source": chunk.source,
            "page": chunk.page,
            "text": chunk.text,
        }
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    result = index.documents.batch_upsert(
        namespace=NAMESPACE,
        documents=documents,
        batch_size=batch_size,
    )
    if result.has_errors:
        details = "; ".join(str(error) for error in result.errors)
        raise RuntimeError(
            f"{result.failed_item_count}/{result.total_item_count} documents failed: {details}"
        )


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Missing {name}. Set it in rag-pdf-warmup/.env")
    return value


def main() -> int:
    from dotenv import load_dotenv
    from openai import OpenAI, OpenAIError
    from pinecone import Pinecone, PineconeError

    load_dotenv(Path(__file__).parent / ".env")
    openai_client = OpenAI(api_key=_require_env("OPENAI_API_KEY"))
    pc = Pinecone(api_key=_require_env("PINECONE_API_KEY"))

    try:
        pages = load_pages(DATA_DIR)
        chunks = chunk_pages(pages)
        for source, count in sorted(Counter(c.source for c in chunks).items()):
            print(f"{source}: {count} chunks")

        vectors = embed_texts(openai_client, [chunk.text for chunk in chunks])
        index = ensure_index(pc)
        upsert_chunks(index, chunks, vectors)
    except OpenAIError as error:
        print(f"Ingest failed: OpenAI error: {error}", file=sys.stderr)
        return 1
    except PineconeError as error:
        print(f"Ingest failed: Pinecone error: {error}", file=sys.stderr)
        return 1
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Ingest failed: {error}", file=sys.stderr)
        return 1

    print(
        f"Upserted {len(chunks)} chunks into {INDEX_NAME!r} (namespace {NAMESPACE!r})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
