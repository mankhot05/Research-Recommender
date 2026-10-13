"""RAG tests. The vector store and LLM are faked or mocked: no network calls."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import rag
from ingest import EMBEDDING_MODEL, NAMESPACE, VECTOR_FIELD
from rag import (
    NO_ANSWER,
    SNIPPET_LENGTH,
    OpenAIChatModel,
    PineconeRetriever,
    RetrievedChunk,
    answer,
    build_messages,
)

CHUNKS = [
    RetrievedChunk(
        source="Attention Is All You Need.pdf",
        page=4,
        text="Multi-head attention runs h attention heads in parallel.",
        score=0.91,
    ),
    RetrievedChunk(
        source="KAN Kolmogorov–Arnold Networks.pdf",
        page=2,
        text="KANs place learnable activations on edges.",
        score=0.74,
    ),
]


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def retrieve(self, question, k):
        self.calls.append((question, k))
        return self.chunks[:k]


class FakeChatModel:
    def __init__(self, reply="Heads run in parallel [1]."):
        self.reply = reply
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages)
        return self.reply


# --- build_messages ---------------------------------------------------------


def test_build_messages_numbers_chunks_in_order_with_source_and_page():
    _, user = build_messages("What is multi-head attention?", CHUNKS)

    content = user["content"]
    first = content.index("[1] (Attention Is All You Need.pdf, p. 4)")
    second = content.index("[2] (KAN Kolmogorov–Arnold Networks.pdf, p. 2)")
    assert first < second
    assert CHUNKS[0].text in content
    assert CHUNKS[1].text in content
    assert "[3]" not in content
    assert content.endswith("Question: What is multi-head attention?")


def test_system_message_states_grounding_rules():
    system, _ = build_messages("q", CHUNKS)

    assert system["role"] == "system"
    assert "ONLY" in system["content"]
    assert "like [2]" in system["content"]
    assert f"reply exactly: {NO_ANSWER}" in system["content"]


# --- answer -----------------------------------------------------------------


def test_answer_passes_k_and_returns_numbered_sources():
    retriever = FakeRetriever(CHUNKS)
    chat_model = FakeChatModel()

    result = answer("  multi-head?  ", k=2, retriever=retriever, chat_model=chat_model)

    assert retriever.calls == [("multi-head?", 2)]
    assert chat_model.calls == [build_messages("multi-head?", CHUNKS)]
    assert result.text == "Heads run in parallel [1]."
    assert [(s.number, s.source, s.page, s.score) for s in result.sources] == [
        (1, "Attention Is All You Need.pdf", 4, 0.91),
        (2, "KAN Kolmogorov–Arnold Networks.pdf", 2, 0.74),
    ]
    assert result.sources[0].snippet == CHUNKS[0].text


def test_answer_uses_default_k_of_4():
    retriever = FakeRetriever(CHUNKS)

    answer("q", retriever=retriever, chat_model=FakeChatModel())

    assert retriever.calls == [("q", 4)]


def test_long_chunk_text_is_truncated_to_snippet_length():
    chunk = RetrievedChunk(source="a.pdf", page=1, text="word\n  " * 200, score=0.5)

    result = answer("q", retriever=FakeRetriever([chunk]), chat_model=FakeChatModel())

    snippet = result.sources[0].snippet
    assert len(snippet) == SNIPPET_LENGTH
    assert snippet.endswith("…")
    assert "\n" not in snippet


def test_no_chunks_returns_i_dont_know_without_calling_llm():
    chat_model = FakeChatModel()

    result = answer("q", retriever=FakeRetriever([]), chat_model=chat_model)

    assert result.text == NO_ANSWER
    assert result.sources == []
    assert chat_model.calls == []


@pytest.mark.parametrize(("question", "k"), [("", 4), ("   ", 4), ("q", 0)])
def test_invalid_input_raises(question, k):
    with pytest.raises(ValueError):
        answer(question, k, retriever=FakeRetriever(CHUNKS), chat_model=FakeChatModel())


def test_missing_chat_model_env_var_is_a_clear_error(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("PINECONE_API_KEY", "pc-test")
    monkeypatch.delenv("OPENAI_CHAT_MODEL", raising=False)

    with pytest.raises(RuntimeError, match="Missing OPENAI_CHAT_MODEL"):
        answer("q", retriever=FakeRetriever(CHUNKS))


# --- PineconeRetriever / OpenAIChatModel with mocked SDK clients -----------


def test_pinecone_retriever_embeds_question_and_maps_matches():
    openai_client = MagicMock()
    openai_client.embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(index=0, embedding=[0.1, 0.2, 0.3])]
    )
    index = MagicMock()
    index.documents.search.return_value = SimpleNamespace(
        matches=[
            SimpleNamespace(
                id="x", score=0.88, source="a.pdf", page=3.0, text="chunk text"
            )
        ]
    )

    chunks = PineconeRetriever(openai_client, index).retrieve("why?", k=3)

    openai_client.embeddings.create.assert_called_once_with(
        model=EMBEDDING_MODEL, input=["why?"]
    )
    kwargs = index.documents.search.call_args.kwargs
    assert kwargs["namespace"] == NAMESPACE
    assert kwargs["top_k"] == 3
    assert kwargs["include_fields"] == ["source", "page", "text"]
    [query] = kwargs["score_by"]
    assert query.field == VECTOR_FIELD
    assert query.values == [0.1, 0.2, 0.3]
    assert chunks == [
        RetrievedChunk(source="a.pdf", page=3, text="chunk text", score=0.88)
    ]


def _chat_response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_openai_chat_model_passes_model_and_messages():
    client = MagicMock()
    client.chat.completions.create.return_value = _chat_response("  answer [1] \n")
    messages = [{"role": "user", "content": "hi"}]

    reply = OpenAIChatModel(client, "test-model").complete(messages)

    client.chat.completions.create.assert_called_once_with(
        model="test-model", messages=messages
    )
    assert reply == "answer [1]"


@pytest.mark.parametrize("content", [None, "", "   "])
def test_openai_chat_model_empty_reply_raises(content):
    client = MagicMock()
    client.chat.completions.create.return_value = _chat_response(content)

    with pytest.raises(RuntimeError, match="empty answer"):
        OpenAIChatModel(client, "test-model").complete([])


# --- CLI ----------------------------------------------------------------------


def test_cli_prints_answer_and_sources(monkeypatch, capsys):
    def fake_answer(question, k):
        assert (question, k) == ("What is attention?", 2)
        return answer(
            question, k, retriever=FakeRetriever(CHUNKS), chat_model=FakeChatModel()
        )

    monkeypatch.setattr(rag, "answer", fake_answer)

    exit_code = rag.main(["What is attention?", "--k", "2"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert out.startswith("Heads run in parallel [1].")
    assert "[1] Attention Is All You Need.pdf, p. 4 (score 0.910)" in out
    assert "[2] KAN Kolmogorov–Arnold Networks.pdf, p. 2 (score 0.740)" in out


def test_cli_reports_errors_and_exits_1(monkeypatch, capsys):
    def failing_answer(question, k):
        raise RuntimeError("Missing OPENAI_CHAT_MODEL. Set it in rag-pdf-warmup/.env")

    monkeypatch.setattr(rag, "answer", failing_answer)

    exit_code = rag.main(["q"])

    assert exit_code == 1
    assert "Query failed: Missing OPENAI_CHAT_MODEL" in capsys.readouterr().err
