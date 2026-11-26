"""Tests for app.llm: offline, with a fake chat model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app import llm
from app.schemas import Critique, PaperVerdict, SearchPlan
from tests.fakes import FakeChatModel, make_paper


def test_get_chat_model_uses_settings_and_temperature_zero(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-test")

    model = llm.get_chat_model()

    assert model.model_name == "gpt-test"
    assert model.temperature == 0


def test_get_chat_model_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(llm.LLMConfigError):
        llm.get_chat_model()


def test_rewrite_queries_returns_structured_plan():
    plan = SearchPlan(queries=['abs:"graph neural network" AND cat:cs.LG'])
    model = FakeChatModel(plan)

    result = llm.rewrite_queries("GNNs for molecules", model=model)

    assert result == plan
    assert model.schemas == [SearchPlan]
    [(system_role, system), (human_role, human)] = model.calls[0]
    assert (system_role, system) == ("system", llm.REWRITE_SYSTEM_PROMPT)
    assert human_role == "human"
    assert "GNNs for molecules" in human
    assert "Feedback" not in human


def test_rewrite_queries_includes_feedback_when_given():
    model = FakeChatModel(SearchPlan(queries=["abs:molecule"]))

    llm.rewrite_queries("GNNs", feedback="nothing on equivariance", model=model)

    human = model.calls[0][1][1]
    assert "nothing on equivariance" in human


def test_search_plan_cleans_and_caps_queries():
    plan = SearchPlan(queries=[" a ", "", "a", "b", "c", "d"])

    assert plan.queries == ["a", "b", "c"]


def test_search_plan_rejects_no_queries():
    with pytest.raises(ValidationError):
        SearchPlan(queries=["  ", ""])


def test_critique_grades_all_papers_in_one_call():
    papers = [make_paper("2401.00001"), make_paper("2401.00002")]
    reply = Critique(
        verdicts=[
            PaperVerdict(arxiv_id="2401.00002", relevant=False, reason="Off topic."),
            PaperVerdict(arxiv_id="2401.00001", relevant=True, reason="Direct hit."),
        ],
        feedback="Missing recent work.",
    )
    model = FakeChatModel(reply)

    result = llm.critique("GNNs for molecules", papers, model=model)

    assert len(model.calls) == 1
    assert model.schemas == [Critique]
    system, human = model.calls[0]
    assert system == ("system", llm.CRITIQUE_SYSTEM_PROMPT)
    assert "GNNs for molecules" in human[1]
    assert "[2401.00001]" in human[1] and "[2401.00002]" in human[1]
    # Output follows input order, not the model's order.
    assert [v.arxiv_id for v in result.verdicts] == ["2401.00001", "2401.00002"]
    assert result.verdicts[0].relevant is True
    assert result.feedback == "Missing recent work."


def test_critique_drops_unknown_fills_missing_and_strips_versions():
    papers = [make_paper("2401.00001"), make_paper("2401.00002")]
    reply = Critique(
        verdicts=[
            PaperVerdict(arxiv_id="2401.00001v3", relevant=True, reason="First."),
            PaperVerdict(arxiv_id="2401.00001", relevant=False, reason="Duplicate."),
            PaperVerdict(arxiv_id="9999.99999", relevant=True, reason="Invented."),
        ],
        feedback="",
    )

    result = llm.critique("q", papers, model=FakeChatModel(reply))

    assert [(v.arxiv_id, v.relevant, v.reason) for v in result.verdicts] == [
        ("2401.00001", True, "First."),
        ("2401.00002", False, llm.NOT_GRADED_REASON),
    ]


def test_critique_with_no_papers_skips_the_llm():
    model = FakeChatModel(None)

    result = llm.critique("q", [], model=model)

    assert result.verdicts == []
    assert result.feedback == llm.NO_PAPERS_FEEDBACK
    assert model.calls == []


def test_paper_verdict_reason_is_one_line():
    verdict = PaperVerdict(arxiv_id="x", relevant=True, reason="Good\n  fit.\n")

    assert verdict.reason == "Good fit."
