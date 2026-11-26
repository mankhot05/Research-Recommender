"""Tests for app.sources: fetch_arxiv with a fake arXiv client, and the paper cache."""

from __future__ import annotations

from datetime import UTC, date, datetime

import arxiv
import pytest

from app import sources
from app.sources import (
    ArxivFetchError,
    append_to_cache,
    fetch_arxiv,
    load_cached_papers,
)
from tests.fakes import make_paper


def make_result(
    entry_id: str = "http://arxiv.org/abs/2401.01234v1",
    title: str = "A Title",
    summary: str = "An abstract.",
    authors: tuple[str, ...] = ("Ada Lovelace",),
    published: datetime = datetime(2024, 1, 3, 12, 0, tzinfo=UTC),
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
    """Stands in for arxiv.Client: records searches and returns canned results.

    `errors` holds one entry per results() call: an exception to raise on that
    call, or None to succeed. Calls past the end of the list succeed.
    """

    def __init__(self, results=(), errors=()):
        self._results = list(results)
        self._errors = list(errors)
        self.page_size = 100
        self.searches: list[arxiv.Search] = []

    def results(self, search: arxiv.Search):
        self.searches.append(search)
        error = self._errors.pop(0) if self._errors else None
        if error is not None:
            raise error
        yield from self._results


def http_error(status: int, start: int = 100) -> arxiv.HTTPError:
    url = f"http://export.arxiv.org/api/query?search_query=x&start={start}&max_results=100"
    return arxiv.HTTPError(url=url, retry=1, status=status)


@pytest.fixture(autouse=True)
def sleeps(monkeypatch) -> list[float]:
    """Record backoff waits instead of really sleeping."""
    waits: list[float] = []
    monkeypatch.setattr(sources, "_sleep", waits.append)
    return waits


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
                    published=datetime(2023, 12, 31, 23, 0, tzinfo=UTC),
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
    assert sources._client.num_retries == 1


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


def test_wraps_non_retryable_errors_without_retrying(use_client, sleeps):
    error = http_error(400, start=0)
    client = use_client(FakeClient(errors=[error]))

    with pytest.raises(ArxivFetchError, match="HTTP 400 on page start=0") as excinfo:
        fetch_arxiv("transformers")

    assert excinfo.value.__cause__ is error
    assert len(client.searches) == 1
    assert sleeps == []


@pytest.mark.parametrize("status", [429, 503])
def test_retries_after_rate_limit_then_succeeds(use_client, sleeps, status):
    client = use_client(FakeClient([make_result()], errors=[http_error(status)]))

    [paper] = fetch_arxiv("transformers")

    assert paper.arxiv_id == "2401.01234"
    assert len(client.searches) == 2
    [wait] = sleeps
    assert 10 <= wait <= 12.5


def test_gives_up_after_repeated_rate_limits(use_client, sleeps):
    errors = [http_error(429) for _ in range(4)]
    client = use_client(FakeClient([make_result()], errors=errors))

    with pytest.raises(ArxivFetchError) as excinfo:
        fetch_arxiv("transformers")

    message = str(excinfo.value)
    assert "HTTP 429 on page start=100" in message
    assert "4 attempts" in message
    assert excinfo.value.__cause__ is errors[-1]
    assert len(client.searches) == 4
    assert len(sleeps) == 3
    for wait, base in zip(sleeps, (10, 30, 90), strict=True):
        assert base <= wait <= base * 1.25


@pytest.mark.parametrize(
    ("max_results", "page_size"), [(7, 7), (200, 200), (2000, 500)]
)
def test_page_size_is_capped_at_500(use_client, max_results, page_size):
    client = use_client(FakeClient())

    fetch_arxiv("transformers", max_results=max_results)

    assert client.page_size == page_size


def test_cache_round_trip(tmp_path):
    path = tmp_path / "papers.jsonl"
    papers = [make_paper("2401.00001"), make_paper("2401.00002", title="Other")]

    assert append_to_cache(papers, path) == 2
    assert load_cached_papers(path) == papers


def test_cache_creates_missing_parent_dirs(tmp_path):
    path = tmp_path / "data" / "nested" / "papers.jsonl"

    append_to_cache([make_paper()], path)

    assert path.exists()


def test_missing_cache_loads_as_empty(tmp_path):
    assert load_cached_papers(tmp_path / "nope.jsonl") == []


def test_cache_skips_known_and_repeated_ids(tmp_path):
    path = tmp_path / "papers.jsonl"
    append_to_cache([make_paper("2401.00001", title="First")], path)

    added = append_to_cache(
        [
            make_paper("2401.00001", title="Changed"),
            make_paper("2401.00002"),
            make_paper("2401.00002v2"),
        ],
        path,
    )

    assert added == 1
    cached = load_cached_papers(path)
    assert [p.arxiv_id for p in cached] == ["2401.00001", "2401.00002"]
    assert cached[0].title == "First"
    assert len(path.read_text().splitlines()) == 2


def test_cache_with_nothing_new_writes_nothing(tmp_path):
    path = tmp_path / "papers.jsonl"

    assert append_to_cache([], path) == 0
    assert not path.exists()


def test_corrupt_cache_line_names_line_number(tmp_path):
    path = tmp_path / "papers.jsonl"
    append_to_cache([make_paper()], path)
    with path.open("a") as file:
        file.write("\n{not json}\n")

    with pytest.raises(ValueError, match=r"papers.jsonl:3"):
        load_cached_papers(path)
