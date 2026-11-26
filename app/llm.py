"""LLM client setup (OpenAI via langchain-openai) and the agent's two LLM steps.

- rewrite_queries: research question (+ optional feedback) -> arXiv search queries.
- critique: grade every retrieved paper in one call, plus feedback on what's missing.

Both use with_structured_output, so the model's reply comes back as a validated
Pydantic object (SearchPlan / Critique) instead of free text we would have to parse.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel

from app.config import get_settings
from app.schemas import Critique, Paper, PaperVerdict, SearchPlan

REWRITE_SYSTEM_PROMPT = """\
You turn a research question into search queries for the arXiv API.
Return 1 to 3 queries. Each query should find a different angle of the question.

arXiv query syntax:
- Field prefixes: ti: (title), abs: (abstract), cat: (category, e.g. cat:cs.LG)
- Boolean operators in capitals: AND, OR, ANDNOT
- Parentheses for grouping, double quotes for exact phrases

Examples:
- abs:"graph neural network" AND abs:molecule
- (ti:"message passing" OR abs:"message passing") AND cat:cs.LG

Prefer specific technical phrases over long natural-language sentences.
If feedback on earlier results is given, change or broaden the queries to fix
what it says is missing. Do not repeat queries that already failed."""

CRITIQUE_SYSTEM_PROMPT = """\
You review search results for a research question.
For EVERY paper listed, return a verdict with:
- arxiv_id: copied exactly from the list
- relevant: true only if the paper directly helps answer the question
- reason: one line on why it is or isn't relevant to THIS question
  (do not just summarize the paper)

Then give overall feedback: which parts of the question the relevant papers
do not cover yet (subtopics, methods, time periods, categories), and how the
search should change to find them."""

NO_PAPERS_FEEDBACK = "No papers were retrieved; broaden the search."
NOT_GRADED_REASON = "Not graded by the model."


class LLMConfigError(RuntimeError):
    """The chat model can't be built because configuration is missing."""


def get_chat_model() -> BaseChatModel:
    """An OpenAI chat model with temperature 0 (as deterministic as possible)."""
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    api_key = settings.openai_api_key
    if api_key is None or not api_key.get_secret_value().strip():
        raise LLMConfigError("Missing OPENAI_API_KEY. Set it in .env")
    return ChatOpenAI(model=settings.openai_chat_model, temperature=0, api_key=api_key)


def rewrite_queries(
    question: str,
    feedback: str | None = None,
    *,
    model: BaseChatModel | None = None,
) -> SearchPlan:
    """Turn a research question into 1–3 arXiv queries, steered by optional feedback."""
    message = f"Research question:\n{question}"
    if feedback:
        message += f"\n\nFeedback on the previous search results:\n{feedback}"
    structured = (model or get_chat_model()).with_structured_output(SearchPlan)
    return structured.invoke([("system", REWRITE_SYSTEM_PROMPT), ("human", message)])


def critique(
    question: str,
    papers: Sequence[Paper],
    *,
    model: BaseChatModel | None = None,
) -> Critique:
    """Grade all papers in one LLM call; returns exactly one verdict per paper."""
    if not papers:
        return Critique(verdicts=[], feedback=NO_PAPERS_FEEDBACK)
    listing = "\n\n".join(_format_paper(paper) for paper in papers)
    message = f"Research question:\n{question}\n\nPapers:\n\n{listing}"
    structured = (model or get_chat_model()).with_structured_output(Critique)
    result = structured.invoke([("system", CRITIQUE_SYSTEM_PROMPT), ("human", message)])
    return Critique(
        verdicts=_align_verdicts(papers, result.verdicts), feedback=result.feedback
    )


def _format_paper(paper: Paper) -> str:
    return (
        f"[{paper.arxiv_id}] {paper.title} ({paper.year}, {paper.primary_category})\n"
        f"{paper.abstract}"
    )


def _align_verdicts(
    papers: Sequence[Paper], verdicts: list[PaperVerdict]
) -> list[PaperVerdict]:
    """One verdict per input paper, in input order.

    LLMs sometimes skip a paper, repeat one, or invent an ID. Unknown IDs are
    dropped, the first verdict for a repeated ID wins, and skipped papers get
    relevant=False so the graph can rely on every paper having a verdict.
    """
    by_id: dict[str, PaperVerdict] = {}
    for verdict in verdicts:
        by_id.setdefault(verdict.arxiv_id, verdict)
    return [
        by_id.get(paper.arxiv_id)
        or PaperVerdict(
            arxiv_id=paper.arxiv_id, relevant=False, reason=NOT_GRADED_REASON
        )
        for paper in papers
    ]
