"""Pydantic models for data that crosses a boundary: the API, arXiv, and LLM output."""

import re
from datetime import date

from pydantic import BaseModel, Field, field_validator, model_validator

_VERSION_SUFFIX = re.compile(r"v\d+$")


def strip_arxiv_version(arxiv_id: str) -> str:
    """Drop the version suffix so 2401.01234v2 becomes 2401.01234."""
    return _VERSION_SUFFIX.sub("", arxiv_id.strip())


class Paper(BaseModel):
    """An arXiv paper's metadata, keyed by its version-less arXiv ID."""

    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    published: date
    year: int
    primary_category: str
    url: str

    @field_validator("arxiv_id", mode="before")
    @classmethod
    def strip_version(cls, value: str) -> str:
        return strip_arxiv_version(value)

    @model_validator(mode="after")
    def check_year_matches_published(self) -> "Paper":
        if self.year != self.published.year:
            raise ValueError(
                f"year {self.year} does not match published date {self.published}"
            )
        return self


MAX_QUERIES = 3


class SearchPlan(BaseModel):
    """LLM output: the arXiv search queries to run for a research question."""

    queries: list[str] = Field(
        description=(
            "1 to 3 arXiv API search queries. Use field prefixes (ti:, abs:, cat:), "
            'AND/OR, parentheses, and quoted phrases, e.g. abs:"graph neural network" '
            "AND cat:cs.LG."
        )
    )

    @field_validator("queries")
    @classmethod
    def clean_queries(cls, queries: list[str]) -> list[str]:
        """Trim, drop blanks and duplicates, and keep at most MAX_QUERIES."""
        cleaned: list[str] = []
        for query in queries:
            query = query.strip()
            if query and query not in cleaned:
                cleaned.append(query)
        if not cleaned:
            raise ValueError("a SearchPlan needs at least one non-empty query")
        return cleaned[:MAX_QUERIES]


class PaperVerdict(BaseModel):
    """LLM output: whether one paper is relevant to the question, and why."""

    arxiv_id: str = Field(description="The paper's arXiv ID, exactly as given.")
    relevant: bool = Field(description="True if the paper helps answer the question.")
    reason: str = Field(
        description="One line on why it is or isn't relevant, specific to the question."
    )

    @field_validator("arxiv_id", mode="before")
    @classmethod
    def strip_version(cls, value: str) -> str:
        return strip_arxiv_version(value)

    @field_validator("reason")
    @classmethod
    def one_line(cls, value: str) -> str:
        """Collapse newlines and repeated spaces so the reason prints on one line."""
        return " ".join(value.split())


class Critique(BaseModel):
    """LLM output: a verdict for every retrieved paper plus overall feedback."""

    verdicts: list[PaperVerdict] = Field(description="One verdict per paper.")
    feedback: str = Field(
        description=(
            "What the results are missing (subtopics, methods, years, categories) "
            "and how to change the search to find it."
        )
    )


class GradedPaper(BaseModel):
    """A retrieved paper after critique: its vector score plus the LLM's verdict."""

    paper: Paper
    score: float
    relevant: bool
    reason: str


class Recommendation(BaseModel):
    """One paper in the final answer, with why it is relevant."""

    paper: Paper
    score: float
    reason: str


class IterationSummary(BaseModel):
    """What one pass of the agent loop searched for and how well it did."""

    iteration: int
    queries: list[str]
    relevant: int
    candidates: int
