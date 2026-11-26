"""Embedding models for papers and queries behind a small Embedder interface.

The implementation is picked at runtime from the EMBEDDER setting (or a name
passed to get_embedder), so a new model only needs a class and a registry entry.

Each embedder also decides how a paper is turned into text (format_paper),
because models are trained on different input formats: OpenAI gets
"title\\n\\nabstract", while SciNCL and SPECTER were trained on
"title[SEP]abstract".
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cache
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from langchain_core.embeddings import Embeddings

from app.config import get_settings
from app.schemas import Paper

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_DIMENSION = 1536


class EmbedderConfigError(RuntimeError):
    """The embedder cannot be built from the current configuration."""


@runtime_checkable
class Embedder(Protocol):
    """Turns papers and queries into fixed-size vectors."""

    name: str
    dimension: int

    def format_paper(self, paper: Paper) -> str: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _check_vectors(
    name: str, dimension: int, vectors: Sequence[Sequence[float]], expected_count: int
) -> list[list[float]]:
    """Fail loudly if a model returned the wrong number or size of vectors."""
    if len(vectors) != expected_count:
        raise RuntimeError(
            f"{name} returned {len(vectors)} embeddings for {expected_count} texts"
        )
    for vector in vectors:
        if len(vector) != dimension:
            raise RuntimeError(
                f"{name} returned a {len(vector)}-d vector, expected {dimension}"
            )
    return [list(vector) for vector in vectors]


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

    def format_paper(self, paper: Paper) -> str:
        """The title, a blank line, then the abstract."""
        return f"{paper.title}\n\n{paper.abstract}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._client.embed_documents(texts)
        return _check_vectors(self.name, self.dimension, vectors, len(texts))

    def embed_query(self, text: str) -> list[float]:
        vector = self._client.embed_query(text)
        return _check_vectors(self.name, self.dimension, [vector], 1)[0]


def _device() -> str:
    """Apple's GPU (mps) when available, otherwise the CPU."""
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


@cache
def _load_sentence_transformer(model_id: str) -> SentenceTransformer:
    """Load a model once per process (the first call downloads it from Hugging Face)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id, device=_device())


class SentenceTransformerEmbedder:
    """A local Hugging Face model run with sentence-transformers.

    Pooling (first/CLS token) and max length (512 tokens) come from each model's
    own sentence-transformers config. Vectors are L2-normalized, so cosine
    similarity equals the dot product.
    """

    name: str
    model_id: str
    dimension = 768
    batch_size = 16

    def __init__(self, model: SentenceTransformer | None = None) -> None:
        # None means "load on first use", so building an embedder downloads nothing.
        self._model = model

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = _load_sentence_transformer(self.model_id)
        return self._model

    def format_paper(self, paper: Paper) -> str:
        """Title and abstract joined by the tokenizer's separator, e.g. [SEP]."""
        return f"{paper.title}{self.model.tokenizer.sep_token}{paper.abstract}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return _check_vectors(
            self.name, self.dimension, self._encode(texts), len(texts)
        )

    def embed_query(self, text: str) -> list[float]:
        # Same encoder as documents: SciNCL and SPECTER are symmetric models.
        return _check_vectors(self.name, self.dimension, self._encode([text]), 1)[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()


class SciNCLEmbedder(SentenceTransformerEmbedder):
    """SciNCL: a BERT model trained on citation-graph neighbours."""

    name = "scincl"
    model_id = "malteos/scincl"


class SpecterEmbedder(SentenceTransformerEmbedder):
    """SPECTER (v1): SciBERT trained so that papers citing each other embed close."""

    name = "specter"
    model_id = "sentence-transformers/allenai-specter"


# Add new implementations here, keyed by their EMBEDDER value.
_EMBEDDERS: dict[str, Callable[[], Embedder]] = {
    "openai": OpenAIEmbedder,
    "scincl": SciNCLEmbedder,
    "specter": SpecterEmbedder,
}


def get_embedder(name: str | None = None) -> Embedder:
    """Build the named embedder, or the one in the EMBEDDER setting (default "openai")."""
    if name is None:
        name = get_settings().embedder
    name = name.strip().lower() or "openai"
    factory = _EMBEDDERS.get(name)
    if factory is None:
        valid = ", ".join(sorted(_EMBEDDERS))
        raise EmbedderConfigError(f"Unknown EMBEDDER {name!r}. Valid values: {valid}")
    return factory()
