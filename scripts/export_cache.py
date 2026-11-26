"""Rebuild the local paper cache (data/papers.jsonl) from a Pinecone index.

Usage: python scripts/export_cache.py [--index papers-openai-1536]

Every record stores its full Paper as metadata, so the index can be read back
into the cache. Papers already cached are kept and not duplicated (by arxiv_id).
Read-only on Pinecone; needs PINECONE_API_KEY but no embedder key.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pinecone import PineconeError

from app.retrieval import RetrievalConfigError, export_papers, open_index
from app.sources import append_to_cache

# The repo-root data/ folder, the same cache ingest_arxiv.py reads and writes.
CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "papers.jsonl"
DEFAULT_INDEX = "papers-openai-1536"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--index",
        default=DEFAULT_INDEX,
        help=f"Pinecone index to export (default {DEFAULT_INDEX})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        index = open_index(args.index)
        papers = export_papers(index)
        added = append_to_cache(papers, CACHE_PATH)
    except (RetrievalConfigError, PineconeError, ValueError, OSError) as error:
        print(f"Export failed: {error}", file=sys.stderr)
        return 1
    print(
        f"Exported {len(papers)} papers from {args.index}; "
        f"cached {added} new in {CACHE_PATH}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
