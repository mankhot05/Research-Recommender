"""Fetches paper metadata and abstracts from external sources such as arXiv.

Fetched papers can also be kept in a local JSON Lines cache (one Paper per line),
so they can be re-ingested later without calling arXiv again.
"""

import logging
import random
import time
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import arxiv
from pydantic import ValidationError

from app.schemas import Paper, strip_arxiv_version

logger = logging.getLogger(__name__)

# One shared client so arXiv's 3-second rate limit applies across all calls.
# The library's own retries re-send with no backoff, so keep them low; rate
# limiting (429/503) is handled by fetch_arxiv's backoff below instead.
_client = arxiv.Client(delay_seconds=3, num_retries=1)

# Big pages mean typical fetches are a single request.
_MAX_PAGE_SIZE = 500
# HTTP statuses that mean "slow down / try again later".
_RETRY_STATUSES = {429, 503}
# Base wait in seconds before each retry of the whole fetch (up to 4 attempts).
_RETRY_WAITS = (10.0, 30.0, 90.0)
# Replaced in tests so they never really sleep.
_sleep = time.sleep


class ArxivFetchError(RuntimeError):
    """Raised when the arXiv API cannot be queried."""


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _to_paper(result: arxiv.Result) -> Paper:
    arxiv_id = strip_arxiv_version(result.get_short_id())
    return Paper(
        arxiv_id=arxiv_id,
        title=_normalize_whitespace(result.title),
        abstract=_normalize_whitespace(result.summary),
        authors=[author.name for author in result.authors],
        published=result.published.date(),
        year=result.published.year,
        primary_category=result.primary_category,
        url=f"https://arxiv.org/abs/{arxiv_id}",
    )


def _describe(error: arxiv.ArxivError) -> str:
    """Say what went wrong and on which page, e.g. 'HTTP 429 on page start=100'."""
    start = parse_qs(urlparse(error.url).query).get("start", ["?"])[0]
    cause = (
        f"HTTP {error.status}" if isinstance(error, arxiv.HTTPError) else error.message
    )
    return f"{cause} on page start={start}"


def _is_retryable(error: arxiv.ArxivError) -> bool:
    return isinstance(error, arxiv.HTTPError) and error.status in _RETRY_STATUSES


def _fetch_once(search: arxiv.Search) -> list[Paper]:
    papers: list[Paper] = []
    seen: set[str] = set()
    for result in _client.results(search):
        paper = _to_paper(result)
        if paper.arxiv_id in seen:
            continue
        seen.add(paper.arxiv_id)
        papers.append(paper)
    return papers


def fetch_arxiv(query: str, max_results: int = 50) -> list[Paper]:
    """Search arXiv and return unique papers in relevance order.

    If arXiv rate-limits us (HTTP 429/503), waits with exponential backoff plus
    jitter and retries the whole fetch, up to len(_RETRY_WAITS) more times.
    """
    if not query.strip():
        raise ValueError("query must not be empty")
    if max_results < 1:
        raise ValueError("max_results must be at least 1")

    _client.page_size = min(max_results, _MAX_PAGE_SIZE)
    search = arxiv.Search(query=query, max_results=max_results)
    attempts = len(_RETRY_WAITS) + 1
    for attempt in range(1, attempts + 1):
        try:
            return _fetch_once(search)
        except arxiv.ArxivError as e:
            if not _is_retryable(e) or attempt == attempts:
                gave_up = f" (gave up after {attempt} attempts)" if attempt > 1 else ""
                raise ArxivFetchError(
                    f"arXiv search failed for query {query!r}: {_describe(e)}{gave_up}"
                ) from e
            base = _RETRY_WAITS[attempt - 1]
            # Jitter spreads out retries so clients don't all come back at once.
            wait = base + random.uniform(0, base * 0.25)
            logger.warning(
                "arXiv returned %s; retrying in %.0fs (attempt %d of %d)",
                _describe(e),
                wait,
                attempt + 1,
                attempts,
            )
            _sleep(wait)
    raise AssertionError("unreachable")


def load_cached_papers(path: Path) -> list[Paper]:
    """Read papers from a JSON Lines cache; a missing file is an empty cache.

    Keeps the first paper for each arxiv_id. Raises ValueError naming the line
    number if a line is not a valid Paper.
    """
    if not path.exists():
        return []
    papers: list[Paper] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as file:
        for lineno, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                paper = Paper.model_validate_json(line)
            except ValidationError as e:
                raise ValueError(f"{path}:{lineno}: not a valid paper: {e}") from e
            if paper.arxiv_id in seen:
                continue
            seen.add(paper.arxiv_id)
            papers.append(paper)
    return papers


def append_to_cache(papers: Iterable[Paper], path: Path) -> int:
    """Append papers whose arxiv_id is not cached yet. Returns how many were added."""
    seen = {paper.arxiv_id for paper in load_cached_papers(path)}
    new: list[Paper] = []
    for paper in papers:
        if paper.arxiv_id in seen:
            continue
        seen.add(paper.arxiv_id)
        new.append(paper)
    if new:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            for paper in new:
                file.write(paper.model_dump_json() + "\n")
    return len(new)
