# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Backend
```bash
cd backend && uvicorn app.main:app --reload --port 8000   # dev server
cd backend && alembic upgrade head                         # run migrations
cd backend && pytest tests/                                # all tests
cd backend && pytest tests/test_rag.py                    # single test file
cd backend && ruff check . --line-length 100               # lint
cd backend && python scripts/batch_ingest_transcripts.py  # bulk ingest .docx transcripts
cd backend && python scripts/seed_transcript.py --prod    # seed JSON transcripts to prod DB
cd backend && python scripts/run_evals.py                 # run agent evals
cd backend && python scripts/run_retrieval_evals.py       # run retrieval evals
```

### Frontend
```bash
cd frontend && npm run dev      # Vite dev server (port 5173)
cd frontend && npm run build    # TypeScript check + Vite build
cd frontend && npm run lint     # ESLint
```

## Architecture

### Request Flow
```
User query (React)
  → POST /agent/query (SSE stream)
  → stream_simple_rag_or_agent()
      1. _resolve_entities_via_llm()   — gpt-4o-mini extracts (ticker, date) pairs
      2. retrieve_relevant_chunks()    — hybrid vector+keyword search (RRF)
      3. OpenAI LLM call              — generates answer with citations
  → SSE "delta" events (tokens) + "done" event (full AgentResponse)
  → Frontend accumulates deltas, renders sources on done
  → Session saved to DB asynchronously (fire-and-forget)
```

### Key Modules

**`app/agents/simple_rag.py`** — Core pipeline. Entity resolution maps natural language → `(ticker, date)` pairs via a cheap LLM call, then resolved against actual call dates in DB. Results cached 24h (`_SCOPE_CACHE`). Supports multi-entity/comparison queries returning multiple pairs.

**`app/rag/retriever.py`** — Hybrid search via Reciprocal Rank Fusion (RRF) merging pgvector cosine similarity + PostgreSQL full-text search. Company list fetched from DB and cached 24h (`_COMPANIES_CACHE`). Cache must be cleared on ingest (done in router) so new companies are immediately recognized.

**`app/rag/ingestion.py`** — Chunks transcripts along speaker turns (not fixed token windows). Each chunk stores speaker/role metadata. First chunk also stores `_full_transcript` for source display.

**`app/agents/streaming.py`** — Wraps the RAG pipeline to yield SSE events. `delta` events carry streamed tokens; `done` carries the full `AgentResponse` with sources and cited spans.

**`app/conversations/`** — Session persistence. Messages are saved fire-and-forget after response completes. Frontend sends `session_id` on every request.

**`app/evals/`** — Eval harness with datasets, retrieval evals, and LLM-as-judge scoring. Uses a separate `eval_document_chunks` table so eval data doesn't pollute production.

### Database
PostgreSQL + pgvector. Two chunk tables: `document_chunks` (production) and `eval_document_chunks` (evals). Migrations in `alembic/versions/`. Schema controlled by `app/models/db_models.py`.

### Ingestion Scripts
- **`batch_ingest_transcripts.py`** — Reads `backend/transcripts/{TICKER}/{YYYY-MM-DD}.docx`, POSTs to `/rag/ingest/manual/upload`. Supports `.docx` and `.txt` (auto-converts). Default target is the production Render URL.
- **`seed_transcript.py`** — Reads JSON files from `backend/eval_data/`, ingests directly via Python (no HTTP). Use `--prod` to target production table.

### Configuration
All settings in `app/config.py` via environment variables. Key defaults:
- `openai_model`: `gpt-4.1-nano` (fast; swap to `gpt-4o-mini` for quality)
- `embedding_model`: `text-embedding-3-small` (1536 dims)
- `default_search_mode`: `hybrid`
- `retrieval_top_k`: 5, `retrieval_threshold`: 0.35
- `chunk_size`: 512 tokens, `chunk_overlap`: 64 tokens

### Caching
Three in-process TTL caches (24h), all in `retriever.py` / `simple_rag.py`:
- `_COMPANIES_CACHE` — known company list from DB; cleared on ingest
- `_SCOPE_CACHE` — entity+date resolution results per query
- `_DATE_RESOLUTION_CACHE` / `_BATCH_DATE_RESOLUTION_CACHE` — call date lookups

On multi-instance deployments each process has its own cache (no distributed invalidation).
