"""Run the agent: recommend arXiv papers for a research question.

Usage: python scripts/recommend.py "message passing GNNs for molecules" [--max-iterations 3] [--no-fetch]
"""

from __future__ import annotations

import argparse
import sys

from openai import OpenAIError
from pinecone import PineconeError

from app.embeddings import EmbedderConfigError
from app.graph import DEFAULT_MAX_ITERATIONS, default_deps, run_graph
from app.llm import LLMConfigError
from app.retrieval import RetrievalConfigError


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question", help="research question")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"most rewrite/search/critique passes (default {DEFAULT_MAX_ITERATIONS})",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="skip arXiv and use only papers already in the index",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    max_iterations = args.max_iterations
    if args.no_fetch and max_iterations > 1:
        # Retrieval always uses the original question, so without new papers
        # every extra pass would return the same top 10 and only cost LLM calls.
        print(
            "--no-fetch: running 1 iteration (the index won't change between passes)."
        )
        max_iterations = 1

    try:
        state = run_graph(
            args.question,
            deps=default_deps(fetch=not args.no_fetch),
            max_iterations=max_iterations,
        )
    except (
        EmbedderConfigError,
        RetrievalConfigError,
        LLMConfigError,
        PineconeError,
        OpenAIError,
        ValueError,
    ) as error:
        print(f"Recommendation failed: {error}", file=sys.stderr)
        return 1

    for step in state["history"]:
        print(
            f"Iteration {step.iteration}: {step.relevant} of {step.candidates} relevant"
        )
        for query in step.queries:
            print(f"  - {query}")

    print()
    if not state["final"]:
        print("No relevant papers found.")
        return 0
    print(f"Top {len(state['final'])}:")
    for rank, rec in enumerate(state["final"], start=1):
        print(f"{rank}. {rec.paper.title} ({rec.paper.year})")
        print(f"   {rec.paper.url}")
        print(f"   {rec.reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
