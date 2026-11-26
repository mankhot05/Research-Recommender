"""Fetch papers from arXiv (or the local cache) and upsert them into Pinecone.

Usage:
  python scripts/ingest_arxiv.py --query "graph neural networks" --max 200
  python scripts/ingest_arxiv.py --from-cache [--embedder openai]

Every arXiv fetch is appended to data/papers.jsonl (de-duplicated by arxiv_id),
so --from-cache can re-ingest those papers, e.g. into another embedder's index,
without calling arXiv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from openai import OpenAIError
from pinecone import PineconeError

from app.embeddings import EmbedderConfigError, get_embedder
from app.retrieval import RetrievalConfigError, get_index, index_name, upsert_papers
from app.schemas import Paper
from app.sources import (
    ArxivFetchError,
    append_to_cache,
    fetch_arxiv,
    load_cached_papers,
)

# The repo-root data/ folder, so the cache is the same from any working directory.
CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "papers.jsonl"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--query", help="arXiv search query")
    source.add_argument(
        "--from-cache",
        action="store_true",
        help="ingest every cached paper instead of calling arXiv",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=200,
        dest="max_results",
        help="max papers (with --query only)",
    )
    parser.add_argument(
        "--embedder",
        default=None,
        help="embedder name for this run (overrides the EMBEDDER setting)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        # Check config and connect before the slow, rate-limited arXiv fetch.
        embedder = get_embedder(args.embedder)
        index = get_index(embedder)
        papers: list[Paper]
        if args.from_cache:
            papers = load_cached_papers(CACHE_PATH)
            if not papers:
                print(
                    f"Ingest failed: no cached papers in {CACHE_PATH}; "
                    "run with --query first",
                    file=sys.stderr,
                )
                return 1
            print(f"Loaded {len(papers)} papers from {CACHE_PATH}")
        else:
            papers = fetch_arxiv(args.query, max_results=args.max_results)
            # Cache before upserting, so the fetch is kept even if the upsert fails.
            added = append_to_cache(papers, CACHE_PATH)
            print(
                f"Fetched {len(papers)} papers for {args.query!r}; "
                f"cached {added} new in {CACHE_PATH}"
            )
        count = upsert_papers(papers, embedder, index=index)
    except (
        EmbedderConfigError,
        RetrievalConfigError,
        ArxivFetchError,
        PineconeError,
        OpenAIError,
        ValueError,
        OSError,
    ) as error:
        print(f"Ingest failed: {error}", file=sys.stderr)
        return 1
    print(f"Upserted {count} papers into {index_name(embedder)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
