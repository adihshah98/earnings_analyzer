# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Backend
```bash
cd backend && uvicorn app.main:app --reload --port 8000        # dev server
cd backend && alembic upgrade head                              # run migrations
cd backend && pytest tests/                                     # all tests
cd backend && pytest tests/test_rag.py                         # single test file
cd backend && ruff check . --line-length 100                    # lint
cd backend && python scripts/batch_ingest_transcripts.py       # bulk ingest .docx transcripts
cd backend && python scripts/seed_transcript.py --prod         # seed JSON transcripts to prod DB
cd backend && python scripts/reextract_financials.py           # re-extract financial summaries
cd backend && python scripts/run_evals.py                      # run agent evals
cd backend && python scripts/run_retrieval_evals.py            # run retrieval evals
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
  → POST /agent/query (SSE stream, session_id in body)
  → stream_simple_rag_or_agent()
      1. _resolve_entities_via_llm()     — gpt-4.1-mini extracts (ticker, date) pairs
                                           from query + last 5 session turns (in-memory buffer)
      2. build_retrieval_plan()          — rewrites query into 2-3 variants, pre-computes embeddings
      3. retrieve_from_plan()            — RRF merge of vector+keyword per pair, injects
                                           financial summary chunks, selects top 3 chunks/pair
      4. OpenAI LLM call (streaming)    — generates answer with [Source N] citations
  → SSE "delta" events (tokens) + "done" event (full AgentResponse with sources)
  → Session turn saved to DB asynchronously (fire-and-forget)
  → In-memory session buffer updated synchronously (for next query)
```

### Key Modules

**`app/agents/simple_rag.py`** — Core pipeline. Entity resolution maps natural language → `(ticker, date)` pairs via a cheap LLM call, resolved against actual call dates in DB. `_SCOPE_CACHE` (24h TTL) memoizes resolution. `build_retrieval_plan()` pre-computes query rewrites and embeddings; `retrieve_from_plan()` executes retrieval with per-pair chunk selection. Supports multi-entity/comparison queries.

**`app/rag/retriever.py`** — Hybrid search via Reciprocal Rank Fusion (RRF, K=60) merging pgvector cosine similarity + PostgreSQL full-text search (`websearch_to_tsquery` with stopword fallback). Metadata filtering by `_ticker_date_pairs` (most specific), `_tickers`, or standard fields. Financial summary chunks injected for all resolved pairs. Company and period lists cached 24h; caches cleared on ingest.

**`app/rag/ingestion.py`** — Speaker-aware transcript chunking (not fixed token windows). Recognizes multiple speaker header formats. Sub-chunks large blocks. Stores speaker/role metadata. Extracts financial summary chunk for each transcript. Computes `doc_id` as `sha256(ticker:call_date).hexdigest()[:16]`.

**`app/rag/fiscal_calendar.py`** — 49 tickers with company-specific fiscal year end months. `parse_fiscal_quarter()`, `compute_period_end()`, and `cy_quarter_label_from_period_end()` map between fiscal labels ("Q3 FY2026") and calendar periods ("Aug–Oct 2025"). Always displayed together in answer prompts.

**`app/agents/streaming.py`** — Wraps the RAG pipeline to yield SSE events. In-memory `_SESSION_RECENT_QUERIES` buffer (TTLCache, 2h TTL, max 5 turns) written synchronously before async DB write — ensures next query sees prior context immediately.

**`app/conversations/`** — Session persistence. `append_conversation_turn()` stores messages fire-and-forget. `get_recent_turns()` extracts last N (user, assistant) pairs for LLM context. History exposed via `GET /conversations/{session_id}/history` with sources.

**`app/auth/`** — Google OAuth 2.0 flow. JWT-based (HS256, 30-day expiry). `get_optional_user()` / `require_user()` FastAPI dependencies. Sessions isolated per `user_id` FK on `conversation_messages`.

**`app/evals/`** — Eval harness using a separate `eval_document_chunks` table (no prod pollution). `use_eval_chunks_context()` is a ContextVar-based async context manager that switches retrieval table. Agent evals (faithfulness/relevance/completeness via LLM judges). Retrieval evals compare vector/keyword/hybrid modes on precision, recall, MRR, and hit rate. Relevance judgments cached by `hash(query + chunk_content)`.

### Database
PostgreSQL + pgvector. Two chunk tables: `document_chunks` (production) and `eval_document_chunks` (evals). Dedicated B-tree columns `company_ticker`, `call_date`, `fiscal_quarter`, `period_end` for filtering (legacy GIN index on metadata JSONB removed). Migrations in `alembic/versions/`. Schema controlled by `app/models/db_models.py`.

**Tables:** `users`, `document_chunks`, `eval_document_chunks`, `conversation_messages`, `eval_results`, `prompt_versions`

### Ingestion Scripts
- **`batch_ingest_transcripts.py`** — Reads `backend/transcripts/{TICKER}/{YYYY-MM-DD}.docx`, POSTs to `/rag/ingest/manual/upload`. Supports `.docx` and `.txt`. Default target is the production Render URL.
- **`seed_transcript.py`** — Reads JSON files from `backend/eval_data/`, ingests directly via Python. Use `--prod` to target production table, else uses eval table.
- **`reextract_financials.py`** — Re-runs financial summary extraction on already-ingested transcripts without full re-ingestion.

### Configuration
All settings in `app/config.py` via environment variables. Key values:
- `openai_model`: `gpt-4.1-mini` (hardcoded; `OPENAI_MODEL` env is ignored)
- `embedding_model`: `text-embedding-3-small` (1536 dims)
- `default_search_mode`: `hybrid`
- `retrieval_top_k`: 20 (pre-RRF candidates per mode); `retrieval_threshold`: 0.3
- `chunk_size`: 512 tokens, `chunk_overlap`: 64 tokens
- `context_token_budget`: 20,000 (max tokens for context in system prompt)
- `agent_temperature`: 0.1
- `admin_api_key`: `""` (empty = no protection in dev)
- `eval_concurrency`: 3

### Caching
Six in-process TTL caches — no distributed invalidation on multi-instance deployments:

| Cache | Location | TTL | Purpose |
|-------|----------|-----|---------|
| `_EMBEDDING_CACHE` | `rag/embeddings.py` | 24h | Dedup OpenAI embedding API calls |
| `_SCOPE_CACHE` | `agents/simple_rag.py` | 24h | Entity + temporal resolution per query |
| `_SESSION_RECENT_QUERIES` | `agents/streaming.py` | 2h | Last 5 turns per session (immediate context) |
| `_COMPANIES_CACHE` | `rag/retriever.py` | 24h | Known company list; cleared on ingest |
| `_PERIODS_CACHE` | `rag/retriever.py` | 24h | ticker → available call dates; cleared on ingest |
| `_relevance_cache` | `evals/retrieval.py` | session | LLM relevance judgments for retrieval evals |

### Warmup
`GET /warmup` pre-warms `_COMPANIES_CACHE` and the OpenAI embedding client. Frontend calls this on mount to absorb cold-start latency (Render free tier). The lifespan handler also attempts warmup on startup; `/warmup` is a fallback.

### Admin Protection
Routes under `/rag/ingest`, `/rag/search`, and all `/evals/*` require `X-Admin-Key: <ADMIN_API_KEY>`. Set `ADMIN_API_KEY=""` in `.env` to disable (dev default).
