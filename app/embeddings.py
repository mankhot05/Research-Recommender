"""Embedding models for papers and queries behind a small Embedder interface.

The implementation is picked at runtime from the EMBEDDER setting, so a new
model (e.g. SPECTER/SciNCL via sentence-transformers) only needs a class and a
registry entry here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from langchain_core.embeddings import Embeddings

from app.config import get_settings
from app.schemas import Paper

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_DIMENSION = 1536


class EmbedderConfigError(RuntimeError):
    """The embedder cannot be built from the current configuration."""


@runtime_checkable
class Embedder(Protocol):
    """Turns documents and queries into fixed-size vectors."""

    name: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class OpenAIEmbedder:
    """OpenAI text-embedding-3-small via langchain-openai."""

    name = "openai"
    model = OPENAI_EMBEDDING_MODEL
    dimension = OPENAI_EMBEDDING_DIMENSION

    def __init__(self, client: Embeddings | None = None) -> None:
        if client is None:
            from langchain_openai import OpenAIEmbeddings

            api_key = get_settings().openai_api_key
            if api_key is None or not api_key.get_secret_value().strip():
                raise EmbedderConfigError("Missing OPENAI_API_KEY. Set it in .env")
            client = OpenAIEmbeddings(
                model=self.model, dimensions=self.dimension, api_key=api_key
            )
        self._client = client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._client.embed_documents(texts)
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"{self.name} returned {len(vectors)} embeddings for {len(texts)} texts"
            )
        for vector in vectors:
            self._check_dimension(vector)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._check_dimension(self._client.embed_query(text))

    def _check_dimension(self, vector: list[float]) -> list[float]:
        if len(vector) != self.dimension:
            raise RuntimeError(
                f"{self.name} returned a {len(vector)}-d vector, expected {self.dimension}"
            )
        return vector


def paper_to_text(paper: Paper) -> str:
    """The text we embed for a paper: its title, a blank line, then its abstract."""
    return f"{paper.title}\n\n{paper.abstract}"


# Add new implementations here, keyed by their EMBEDDER value.
_EMBEDDERS: dict[str, Callable[[], Embedder]] = {
    "openai": OpenAIEmbedder,
}


def get_embedder() -> Embedder:
    """Build the embedder named by the EMBEDDER setting (default "openai")."""
    name = get_settings().embedder.strip().lower() or "openai"
    factory = _EMBEDDERS.get(name)
    if factory is None:
        valid = ", ".join(sorted(_EMBEDDERS))
        raise EmbedderConfigError(f"Unknown EMBEDDER {name!r}. Valid values: {valid}")
    return factory()
