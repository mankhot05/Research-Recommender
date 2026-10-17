"""Pydantic request/response models for the API."""

import re
from datetime import date

from pydantic import BaseModel, field_validator, model_validator

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
