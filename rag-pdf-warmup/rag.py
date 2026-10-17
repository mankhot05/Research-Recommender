"""Answer questions from the PDFs in the `pdf-warmup` Pinecone index.

Pipeline: embed question -> retrieve top-k chunks -> prompt the chat model to
answer only from those numbered chunks, citing them like [2] -> return the
answer with its sources.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from ingest import INDEX_NAME, NAMESPACE, VECTOR_FIELD, embed_texts, require_env

if TYPE_CHECKING:
    from openai import OpenAI
    from pinecone.index import Index

DEFAULT_K = 4
SNIPPET_LENGTH = 300
NO_ANSWER = "I don't know"
MAX_RETRIES = 3
REQUEST_TIMEOUT_SECONDS = 30.0

SYSTEM_PROMPT = f"""You answer questions about research papers using ONLY the numbered \
context chunks provided by the user.

Rules:
- Use only information stated in the context chunks. Do not use prior knowledge.
- Cite every claim with the number of the chunk that supports it, like [2]. \
Cite several chunks like [1][3].
- Only cite chunk numbers that appear in the context.
- If the context does not contain the answer, reply exactly: {NO_ANSWER}"""


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by the vector store, with its similarity score."""

    source: str
    page: int
    text: str
    score: float


@dataclass(frozen=True)
class Source:
    """A cited chunk as shown to the user. `number` matches the [n] in the answer."""

    number: int
    source: str
    page: int
    snippet: str
    score: float


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Source]


class Retriever(Protocol):
    def retrieve(self, question: str, k: int) -> list[RetrievedChunk]: ...


class ChatModel(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


class PineconeRetriever:
    """Embeds the question with the ingest model and runs a dense-vector search."""

    def __init__(self, openai_client: OpenAI, index: Index) -> None:
        self._openai_client = openai_client
        self._index = index

    def retrieve(self, question: str, k: int) -> list[RetrievedChunk]:
        from pinecone import DenseVectorQuery

        [query_vector] = embed_texts(self._openai_client, [question])
        response = self._index.documents.search(
            namespace=NAMESPACE,
            top_k=k,
            score_by=[DenseVectorQuery(field=VECTOR_FIELD, values=query_vector)],
            include_fields=["source", "page", "text"],
        )
        return [
            RetrievedChunk(
                source=match.source,
                page=int(match.page),
                text=match.text,
                score=float(match.score),
            )
            for match in response.matches
        ]


class OpenAIChatModel:
    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        content = response.choices[0].message.content if response.choices else None
        if not content or not content.strip():
            raise RuntimeError(f"Chat model {self._model!r} returned an empty answer")
        return content.strip()


def build_messages(
    question: str, chunks: Sequence[RetrievedChunk]
) -> list[dict[str, str]]:
    """Number chunks [1]..[k] in retrieval order and wrap them with the rules."""
    context = "\n\n".join(
        f"[{number}] ({chunk.source}, p. {chunk.page})\n{chunk.text}"
        for number, chunk in enumerate(chunks, start=1)
    )
    user = f"Context:\n\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _snippet(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= SNIPPET_LENGTH:
        return collapsed
    return collapsed[: SNIPPET_LENGTH - 1].rstrip() + "…"


def _to_sources(chunks: Sequence[RetrievedChunk]) -> list[Source]:
    return [
        Source(
            number=number,
            source=chunk.source,
            page=chunk.page,
            snippet=_snippet(chunk.text),
            score=chunk.score,
        )
        for number, chunk in enumerate(chunks, start=1)
    ]


def build_dependencies() -> tuple[Retriever, ChatModel]:
    """Build the real Pinecone retriever and OpenAI chat model from env.

    Both SDKs retry transient failures (429/5xx/timeouts) with backoff; the
    policy is set explicitly here rather than wrapped in a second retry layer.
    """
    from dotenv import load_dotenv
    from openai import OpenAI
    from pinecone import Pinecone, RetryConfig

    load_dotenv(Path(__file__).parent / ".env")
    openai_client = OpenAI(
        api_key=require_env("OPENAI_API_KEY"),
        max_retries=MAX_RETRIES,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    chat_model_name = require_env("OPENAI_CHAT_MODEL")
    pc = Pinecone(
        api_key=require_env("PINECONE_API_KEY"),
        timeout=REQUEST_TIMEOUT_SECONDS,
        retry_config=RetryConfig(max_retries=MAX_RETRIES),
    )
    index = pc.index(INDEX_NAME)
    return (
        PineconeRetriever(openai_client, index),
        OpenAIChatModel(openai_client, chat_model_name),
    )


def answer(
    question: str,
    k: int = DEFAULT_K,
    *,
    retriever: Retriever | None = None,
    chat_model: ChatModel | None = None,
) -> Answer:
    """Answer `question` from the top-`k` chunks, returning cited sources.

    `retriever` and `chat_model` default to Pinecone and OpenAI built from env;
    pass your own to test or swap backends.
    """
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")

    if retriever is None or chat_model is None:
        default_retriever, default_chat_model = build_dependencies()
        retriever = retriever or default_retriever
        chat_model = chat_model or default_chat_model

    chunks = retriever.retrieve(question, k)
    if not chunks:
        # Nothing to ground an answer in; skip the LLM call entirely.
        return Answer(text=NO_ANSWER, sources=[])

    text = chat_model.complete(build_messages(question, chunks))
    return Answer(text=text, sources=_to_sources(chunks))


def _print_answer(result: Answer) -> None:
    print(result.text)
    if result.sources:
        print("\nSources:")
    for source in result.sources:
        print(
            f"[{source.number}] {source.source}, p. {source.page} "
            f"(score {source.score:.3f})\n    {source.snippet}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    from openai import OpenAIError
    from pinecone import PineconeError

    parser = argparse.ArgumentParser(description="Ask a question about the PDFs.")
    parser.add_argument("question", help="Question to answer from the PDFs")
    parser.add_argument(
        "--k", type=int, default=DEFAULT_K, help="Number of chunks to retrieve"
    )
    args = parser.parse_args(argv)

    try:
        result = answer(args.question, k=args.k)
    except OpenAIError as error:
        print(f"Query failed: OpenAI error: {error}", file=sys.stderr)
        return 1
    except PineconeError as error:
        print(f"Query failed: Pinecone error: {error}", file=sys.stderr)
        return 1
    except (RuntimeError, ValueError) as error:
        print(f"Query failed: {error}", file=sys.stderr)
        return 1

    _print_answer(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
