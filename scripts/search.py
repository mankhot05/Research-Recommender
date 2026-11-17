"""Search the embedder's Pinecone index for papers relevant to a question.

Usage: python scripts/search.py "message passing for molecules" --k 5 --year-min 2022
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
from openai import OpenAIError
from pinecone import PineconeError

from app.embeddings import EmbedderConfigError
from app.retrieval import RetrievalConfigError, search


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question", help="research question")
    parser.add_argument("--k", type=int, default=5, help="number of results")
    parser.add_argument("--year-min", type=int, help="only papers from this year on")
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        help="arXiv primary category, e.g. cs.LG (repeatable)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv()
    try:
        results = search(
            args.question,
            k=args.k,
            year_min=args.year_min,
            categories=args.categories,
        )
    except (
        EmbedderConfigError,
        RetrievalConfigError,
        PineconeError,
        OpenAIError,
        ValueError,
    ) as error:
        print(f"Search failed: {error}", file=sys.stderr)
        return 1

    if not results:
        print("No matching papers.")
    for rank, (paper, score) in enumerate(results, start=1):
        print(
            f"{rank}. [{score:.3f}] {paper.year} {paper.primary_category}  {paper.title}"
        )
        print(f"   {paper.url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
