# rag-pdf-warmup

Question answering over the PDFs in `data/`, with cited sources.

- `ingest.py` loads each PDF page, splits it into chunks, embeds the chunks with OpenAI `text-embedding-3-small`, and upserts them into the Pinecone serverless index `pdf-warmup`.
- `rag.py` has `answer(question, k=4)`. It retrieves the top-k chunks and has an OpenAI chat model answer only from them, citing chunks like `[2]`. It also works from the command line: `python rag.py "question"`.
- `api.py` serves `answer()` over HTTP with FastAPI.

## Running the API

### 1. Set up (Python 3.12)

```bash
cd rag-pdf-warmup
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2. Configure `.env`

```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
OPENAI_CHAT_MODEL=<an OpenAI chat model id>
```

### 3. Ingest the PDFs

Put PDFs in `data/`, then run:

```bash
.venv/bin/python ingest.py
```

Re-running is safe. Chunk IDs are deterministic, so existing chunks are overwritten instead of duplicated.

### 4. Start the server

```bash
.venv/bin/uvicorn api:app --reload
```

The interactive docs are at http://127.0.0.1:8000/docs.

### 5. Call it

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}

curl -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is multi-head attention?", "k": 4}'
```

```json
{
  "answer": "Multi-head attention runs several attention heads in parallel [1]...",
  "sources": [
    {"number": 1, "source": "Attention Is All You Need.pdf", "page": 4,
     "snippet": "Instead of performing a single attention function...", "score": 0.83}
  ]
}
```

| Endpoint | Request | Responses |
|---|---|---|
| `GET /health` | none | `200 {"status": "ok"}`. Liveness only, with no external calls. |
| `POST /query` | `{"question": str (1-2000 chars), "k": int (1-20, default 4)}` | `200 {answer, sources[]}`, `422` for an invalid body, `502` when OpenAI or Pinecone fails, `503` when configuration is missing |

Each `[n]` in `answer` matches `sources[n-1].number`. When the retrieved chunks don't contain the answer, the response is `"I don't know"`.

**Tracing.** Send an `X-Request-ID` header and it is echoed back as `X-Trace-ID`. If you don't send one, the server generates one. Server logs are JSON lines tagged with the same `trace_id`.

### Tests

```bash
.venv/bin/pytest
```

The tests mock OpenAI, Pinecone and `answer()`, so they need no keys and make no network calls.
