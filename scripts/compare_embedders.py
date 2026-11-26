"""Compare the top-k papers each embedder retrieves for the same queries.

Usage:
  python scripts/compare_embedders.py                      # queries from docs/benchmark_queries.txt
  python scripts/compare_embedders.py "protein folding" --k 3 --embedder openai --embedder scincl

Prints one markdown table per query (one column per embedder) and writes them to
docs/embedding_comparison.md. Each embedder searches its own index, which must
already be ingested. Rankings are compared, not scores: cosine scores from
different models are on different scales.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from openai import OpenAIError
from pinecone import PineconeError

from app.embeddings import EmbedderConfigError, get_embedder
from app.retrieval import RetrievalConfigError, get_index, search
from app.schemas import Paper

DOCS = Path(__file__).resolve().parent.parent / "docs"
DEFAULT_QUERIES_FILE = DOCS / "benchmark_queries.txt"
DEFAULT_OUTPUT = DOCS / "embedding_comparison.md"
DEFAULT_EMBEDDERS = ["openai", "scincl", "specter"]

SCORE_NOTE = (
    "Each cell is `title (score)`. Scores are cosine similarity within one model; "
    "don't compare them across columns, only the rankings."
)

# query -> embedder name -> [(paper, score), ...] best first
Results = dict[str, dict[str, list[tuple[Paper, float]]]]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "queries", nargs="*", help="queries (default: read from --queries-file)"
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=DEFAULT_QUERIES_FILE,
        help="one query per line; blank lines and # comments are skipped",
    )
    parser.add_argument("--k", type=int, default=5, help="results per embedder")
    parser.add_argument(
        "--embedder",
        action="append",
        dest="embedders",
        help=f"embedder to compare (repeatable; default: {', '.join(DEFAULT_EMBEDDERS)})",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUTPUT, help="markdown output file"
    )
    return parser.parse_args(argv)


def load_queries(path: Path) -> list[str]:
    """Non-empty, non-comment lines of the file, stripped."""
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    queries = [line for line in lines if line and not line.startswith("#")]
    if not queries:
        raise ValueError(f"no queries in {path}")
    return queries


def run_comparison(queries: list[str], embedder_names: list[str], k: int) -> Results:
    """Search every query with every embedder, each against its own index."""
    results: Results = {query: {} for query in queries}
    # Embedders on the outside, so each model and index is set up only once.
    for name in embedder_names:
        embedder = get_embedder(name)
        index = get_index(embedder, create=False)
        for query in queries:
            results[query][name] = search(query, k, embedder=embedder, index=index)
    return results


def _cell(hit: tuple[Paper, float] | None) -> str:
    if hit is None:
        return "—"
    paper, score = hit
    title = paper.title.replace("|", "\\|")
    return f"{title} ({score:.3f})"


def format_report(results: Results, embedder_names: list[str], k: int) -> str:
    """A markdown document with one rank-by-embedder table per query."""
    lines = ["# Embedding comparison", "", SCORE_NOTE]
    for number, (query, by_embedder) in enumerate(results.items(), start=1):
        lines += [
            "",
            f"## {number}. {query}",
            "",
            "| Rank | " + " | ".join(embedder_names) + " |",
            "|---" * (len(embedder_names) + 1) + "|",
        ]
        for rank in range(1, k + 1):
            cells = []
            for name in embedder_names:
                hits = by_embedder.get(name, [])
                cells.append(_cell(hits[rank - 1] if rank <= len(hits) else None))
            lines.append(f"| {rank} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    embedder_names = args.embedders or DEFAULT_EMBEDDERS
    try:
        queries = args.queries or load_queries(args.queries_file)
        results = run_comparison(queries, embedder_names, args.k)
        report = format_report(results, embedder_names, args.k)
        print(report)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
    except (
        EmbedderConfigError,
        RetrievalConfigError,
        PineconeError,
        OpenAIError,
        ValueError,
        OSError,
    ) as error:
        print(f"Comparison failed: {error}", file=sys.stderr)
        return 1
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
