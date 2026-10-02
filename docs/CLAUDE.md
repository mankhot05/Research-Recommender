# Research Paper Recommender

Agentic RAG app: a research question goes in, and a ranked list of arXiv papers comes out, each with a one-line reason it is relevant.
Stack: Python 3.12, LangChain + LangGraph, Pinecone, Hugging Face sentence-transformers (SPECTER), OpenAI, FastAPI, Docker.
The roadmap and current chunk are in @docs/ROADMAP.md.

## Commands
- Install: `pip install -r requirements.txt`
- Test: `pytest -q`
- Run the API (from Chunk 5 on): `uvicorn app.main:app --reload`

## Conventions
- App code goes in `app/` and tests go in `tests/`. One module per concern: main, graph, retrieval, embeddings, sources, llm, schemas.
- Use type hints everywhere. Use Pydantic models for any data that crosses a boundary (API, LLM output, external APIs).
- Read secrets only from env vars loaded from `.env`. Never commit `.env`, and keep `.env.example` up to date.
- Keep changes small and focused. Add or update a test for every behavior change, and run `pytest -q` before saying a task is done.

## Working with me (I'm learning)
- I'm using this project to learn AI engineering. After each change, give me 3–5 bullets on what you did and why, and name the key concepts.
- Prefer simple, readable code over clever code. Ask before adding a new dependency.
- Don't run `git commit`. I review the diff and commit myself.
