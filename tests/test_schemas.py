"""Tests for the Paper schema's validation rules."""

from datetime import date

import pytest
from pydantic import ValidationError

from app.schemas import Paper


def paper_kwargs(**overrides) -> dict:
    kwargs = {
        "arxiv_id": "2401.01234",
        "title": "A Title",
        "abstract": "An abstract.",
        "authors": ["Ada Lovelace"],
        "published": date(2024, 1, 3),
        "year": 2024,
        "primary_category": "cs.LG",
        "url": "https://arxiv.org/abs/2401.01234",
    }
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2401.01234v2", "2401.01234"),
        ("2401.01234v12", "2401.01234"),
        ("2401.01234", "2401.01234"),
        ("hep-th/9901001v1", "hep-th/9901001"),
    ],
)
def test_arxiv_id_version_is_stripped(raw, expected):
    assert Paper(**paper_kwargs(arxiv_id=raw)).arxiv_id == expected


def test_year_must_match_published():
    with pytest.raises(ValidationError, match="does not match"):
        Paper(**paper_kwargs(year=2023))
