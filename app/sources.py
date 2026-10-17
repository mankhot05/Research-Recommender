"""Fetches paper metadata and abstracts from external sources such as arXiv."""

import arxiv

from app.schemas import Paper, strip_arxiv_version

# One shared client so arXiv's 3-second rate limit applies across all calls.
_client = arxiv.Client(delay_seconds=3, num_retries=3)


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


def fetch_arxiv(query: str, max_results: int = 50) -> list[Paper]:
    """Search arXiv and return unique papers in relevance order."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if max_results < 1:
        raise ValueError("max_results must be at least 1")

    search = arxiv.Search(query=query, max_results=max_results)
    papers: list[Paper] = []
    seen: set[str] = set()
    try:
        for result in _client.results(search):
            paper = _to_paper(result)
            if paper.arxiv_id in seen:
                continue
            seen.add(paper.arxiv_id)
            papers.append(paper)
    except arxiv.ArxivError as e:
        raise ArxivFetchError(f"arXiv search failed for query {query!r}") from e
    return papers
