"""Fetch papers from arXiv and upsert them into the embedder's Pinecone index.

Usage: python scripts/ingest_arxiv.py --query "graph neural networks" --max 200
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
from openai import OpenAIError
from pinecone import PineconeError

from app.embeddings import EmbedderConfigError, get_embedder
from app.retrieval import RetrievalConfigError, get_index, index_name, upsert_papers
from app.sources import ArxivFetchError, fetch_arxiv


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--query", required=True, help="arXiv search query")
    parser.add_argument(
        "--max", type=int, default=200, dest="max_results", help="max papers"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv()
    try:
        # Check config and connect before the slow, rate-limited arXiv fetch.
        embedder = get_embedder()
        index = get_index(embedder)
        papers = fetch_arxiv(args.query, max_results=args.max_results)
        print(f"Fetched {len(papers)} papers for {args.query!r}")
        count = upsert_papers(papers, embedder, index=index)
    except (
        EmbedderConfigError,
        RetrievalConfigError,
        ArxivFetchError,
        PineconeError,
        OpenAIError,
        ValueError,
    ) as error:
        print(f"Ingest failed: {error}", file=sys.stderr)
        return 1
    print(f"Upserted {count} papers into {index_name(embedder)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
