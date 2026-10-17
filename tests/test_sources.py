"""Tests for app.sources.fetch_arxiv using a fake arXiv client (no network)."""

from __future__ import annotations

from datetime import date, datetime, timezone

import arxiv
import pytest

from app import sources
from app.sources import ArxivFetchError, fetch_arxiv


def make_result(
    entry_id: str = "http://arxiv.org/abs/2401.01234v1",
    title: str = "A Title",
    summary: str = "An abstract.",
    authors: tuple[str, ...] = ("Ada Lovelace",),
    published: datetime = datetime(2024, 1, 3, 12, 0, tzinfo=timezone.utc),
    primary_category: str = "cs.LG",
) -> arxiv.Result:
    return arxiv.Result(
        entry_id=entry_id,
        title=title,
        summary=summary,
        authors=[arxiv.Result.Author(name) for name in authors],
        published=published,
        primary_category=primary_category,
    )


class FakeClient:
    """Stands in for arxiv.Client: records searches and returns canned results."""

    def __init__(self, results=(), error: Exception | None = None):
        self._results = list(results)
        self._error = error
        self.searches: list[arxiv.Search] = []

    def results(self, search: arxiv.Search):
        self.searches.append(search)
        if self._error is not None:
            raise self._error
        yield from self._results


@pytest.fixture
def use_client(monkeypatch):
    def _install(client: FakeClient) -> FakeClient:
        monkeypatch.setattr(sources, "_client", client)
        return client

    return _install


def test_strips_version_suffix(use_client):
    use_client(FakeClient([make_result(entry_id="http://arxiv.org/abs/2401.01234v2")]))

    [paper] = fetch_arxiv("transformers")

    assert paper.arxiv_id == "2401.01234"
    assert paper.url == "https://arxiv.org/abs/2401.01234"


def test_strips_version_suffix_from_old_style_id(use_client):
    use_client(
        FakeClient([make_result(entry_id="http://arxiv.org/abs/hep-th/9901001v1")])
    )

    [paper] = fetch_arxiv("string theory")

    assert paper.arxiv_id == "hep-th/9901001"


def test_normalizes_whitespace_in_title_and_abstract(use_client):
    use_client(
        FakeClient(
            [
                make_result(
                    title="  Attention Is\n  All You\tNeed ",
                    summary="\nWe propose  a new\n\nnetwork architecture.  ",
                )
            ]
        )
    )

    [paper] = fetch_arxiv("attention")

    assert paper.title == "Attention Is All You Need"
    assert paper.abstract == "We propose a new network architecture."


def test_deduplicates_by_arxiv_id_keeping_first(use_client):
    use_client(
        FakeClient(
            [
                make_result(
                    entry_id="http://arxiv.org/abs/2401.01234v2", title="First"
                ),
                make_result(
                    entry_id="http://arxiv.org/abs/2402.05678v1", title="Other"
                ),
                make_result(entry_id="http://arxiv.org/abs/2401.01234v1", title="Dup"),
            ]
        )
    )

    papers = fetch_arxiv("anything")

    assert [p.arxiv_id for p in papers] == ["2401.01234", "2402.05678"]
    assert papers[0].title == "First"


def test_maps_metadata_fields(use_client):
    use_client(
        FakeClient(
            [
                make_result(
                    authors=("Ada Lovelace", "Alan Turing"),
                    published=datetime(2023, 12, 31, 23, 0, tzinfo=timezone.utc),
                    primary_category="stat.ML",
                )
            ]
        )
    )

    [paper] = fetch_arxiv("anything")

    assert paper.authors == ["Ada Lovelace", "Alan Turing"]
    assert paper.published == date(2023, 12, 31)
    assert paper.year == 2023
    assert paper.primary_category == "stat.ML"


def test_passes_query_and_max_results_to_search(use_client):
    client = use_client(FakeClient())

    assert fetch_arxiv("graph neural networks", max_results=7) == []

    [search] = client.searches
    assert search.query == "graph neural networks"
    assert search.max_results == 7


def test_shared_client_respects_rate_limit_settings():
    assert isinstance(sources._client, arxiv.Client)
    assert sources._client.delay_seconds == 3
    assert sources._client.num_retries == 3


@pytest.mark.parametrize(
    ("query", "max_results"),
    [("", 10), ("   ", 10), ("transformers", 0)],
)
def test_rejects_invalid_arguments_without_calling_client(
    use_client, query, max_results
):
    client = use_client(FakeClient())

    with pytest.raises(ValueError):
        fetch_arxiv(query, max_results=max_results)

    assert client.searches == []


def test_wraps_arxiv_errors(use_client):
    error = arxiv.HTTPError(
        url="http://export.arxiv.org/api/query", retry=3, status=503
    )
    use_client(FakeClient(error=error))

    with pytest.raises(ArxivFetchError) as excinfo:
        fetch_arxiv("transformers")

    assert excinfo.value.__cause__ is error
