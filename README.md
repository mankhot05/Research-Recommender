# Research Paper Recommender

An agentic RAG app: a research question goes in, and a ranked list of arXiv papers comes out, each with a one-line reason it is relevant.

Stack: Python 3.12, LangChain + LangGraph, Pinecone, sentence-transformers (SPECTER), OpenAI, FastAPI.

## Setup

Requires Python 3.12+.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # then fill in OPENAI_API_KEY and PINECONE_API_KEY
```

`pip install -e .` installs the `app` package in editable mode, so code changes take effect without reinstalling. The `[dev]` extra adds pytest and ruff.

## Usage

Run from the repo root.

```bash
# Fetch papers from arXiv, embed them and upsert them into Pinecone
python scripts/ingest_arxiv.py --query "graph neural networks" --max 200

# Search the index
python scripts/search.py "message passing for molecules" --k 5 --year-min 2022
python scripts/search.py "protein folding" --category q-bio.BM --category cs.LG
```

The index is named after the embedder (for example `papers-openai-1536`) and is created on first ingest. Set `EMBEDDER` in `.env` to choose the embedding model (default `openai`).

## Development

```bash
pytest -q
ruff check app tests scripts && ruff format app tests scripts
```
