# Research Paper Recommender — Roadmap

## Working agreement
- I code with **Claude Code on my Mac**. A Claude project ("Research Paper Recommender") is my tech lead and tutor: it plans each chunk, writes the Claude Code prompts and reviews the output.
- Pace: as fast as possible. Learning comes mainly from outside resources (GitHub repos, Medium articles, papers) plus reviewing diffs.
- Every chunk has four steps:
  1. **Learn:** curated links.
  2. **Build:** with Claude Code.
  3. **Review:** walk through the diff. I must be able to explain every file.
  4. **Interview:** 3–5 practice questions.

## Chunks
- [ ] **0. Setup:** install Claude Code, get API keys (OpenAI, Pinecone, HF), create the repo, CLAUDE.md, plan-mode workflow, scaffold.
- [ ] **1. Warm-up, Chat-with-PDF** (separate small repo): chunk → embed → Pinecone → retrieve → answer with citations, plus a FastAPI `/query` endpoint.
- [ ] **2. arXiv retrieval core:** fetch abstracts, upsert them with `arxiv_id` metadata, vector search script.
- [ ] **3. SPECTER/SciNCL embeddings:** domain-specific embeddings via sentence-transformers.
- [ ] **4. LangGraph agent loop:** rewrite → search → retrieve → critique → loop or finish.
- [ ] **5. FastAPI `/recommend`:** Pydantic request/response models.
- [ ] **6. Docker + portfolio polish:** Dockerfile, plus a README with an architecture diagram and example queries.
- [ ] **7. (Optional) Evaluation:** hybrid BM25 + reranking, and a small labeled set or RAGAS.

## Current chunk
Chunk 0: Setup

## Target file layout
```
research-recommender/
├── app/  main.py  graph.py  retrieval.py  embeddings.py  sources.py  llm.py  schemas.py
├── tests/
├── docs/ROADMAP.md
├── .env.example  requirements.txt  Dockerfile  README.md  CLAUDE.md
```
