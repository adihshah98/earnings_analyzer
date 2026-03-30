# Architecture — Earnings Analyzer

Deep technical documentation for the Earnings Analyzer RAG system.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Request Pipeline](#2-request-pipeline)
3. [Entity & Temporal Resolution](#3-entity--temporal-resolution)
4. [Retrieval Engine](#4-retrieval-engine)
5. [Ingestion Pipeline](#5-ingestion-pipeline)
6. [Fiscal Calendar System](#6-fiscal-calendar-system)
7. [Streaming & SSE Protocol](#7-streaming--sse-protocol)
8. [Conversation & Session Management](#8-conversation--session-management)
9. [Caching Architecture](#9-caching-architecture)
10. [Auth System](#10-auth-system)
11. [Eval Framework](#11-eval-framework)
12. [Database Schema](#12-database-schema)
13. [Configuration Reference](#13-configuration-reference)

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    React Frontend                        │
│         (Vite + TypeScript, port 5173)                  │
└────────────────────────┬────────────────────────────────┘
                         │ POST /agent/query (SSE)
                         │ session_id, query, search_mode
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  FastAPI Backend                         │
│                   (port 8000)                           │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │            stream_simple_rag_or_agent()           │  │
│  │                                                  │  │
│  │  1. Entity resolution (gpt-4.1-mini)             │  │
│  │  2. Retrieval (pgvector + full-text, RRF)        │  │
│  │  3. Generation (gpt-4.1-mini, streaming)         │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ Conversations│  │  Eval Harness │  │  Auth (OAuth) │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
└────────────────────────┬────────────────────────────────┘
                         │ asyncpg
                         ▼
┌─────────────────────────────────────────────────────────┐
│           PostgreSQL + pgvector                         │
│                                                         │
│  document_chunks      eval_document_chunks              │
│  conversation_messages   users   eval_results           │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Request Pipeline

`app/agents/simple_rag.py` → `stream_simple_rag_or_agent()`

```
query + session_id + search_mode
         │
         ▼
  ┌─────────────────┐
  │ Greeting check  │ → short-circuit with static response
  └────────┬────────┘
           │
           ▼
  ┌─────────────────────────────┐
  │ Load session context        │  last 5 turns from in-memory buffer
  │ (TTLCache, 2h, 2000 slots)  │  (or DB fallback if cache cold)
  └────────┬────────────────────┘
           │
           ▼
  ┌──────────────────────────────────┐
  │ _resolve_entities_via_llm()      │  gpt-4.1-mini, JSON output
  │                                  │  → [(ticker, call_date), ...]
  │ Input:  query + last 5 turns     │  cached in _SCOPE_CACHE (24h)
  │ Output: scope dict with pairs    │
  └────────┬─────────────────────────┘
           │
           ▼
  ┌──────────────────────────────────┐
  │ build_retrieval_plan()           │
  │  - rewrite query → 2-3 variants  │  e.g. "revenue" →
  │  - pre-compute embeddings        │    ["actual revenue",
  │  - resolve dates vs DB           │     "guided revenue",
  │                                  │     "revenue growth"]
  └────────┬─────────────────────────┘
           │
           ▼
  ┌──────────────────────────────────────────────────────┐
  │ retrieve_from_plan()                                 │
  │  For each (ticker, date) pair:                       │
  │    - vector search (pgvector cosine, top-20)         │
  │    - keyword search (PostgreSQL full-text, top-20)   │
  │    - RRF merge → top-N unified                       │
  │    - select top-3 chunks per pair                    │
  │  + inject financial summary chunks for each pair     │
  └────────┬─────────────────────────────────────────────┘
           │
           ▼
  ┌──────────────────────────────────┐
  │ Build system prompt              │
  │  - formatted sources with labels │
  │  - fiscal/calendar period context│
  │  - citation instructions         │
  └────────┬─────────────────────────┘
           │
           ▼
  ┌──────────────────────────────────┐
  │ OpenAI streaming call            │
  │ gpt-4.1-mini, temp=0.1           │
  │ Yields delta tokens → SSE        │
  └────────┬─────────────────────────┘
           │
           ▼
  ┌──────────────────────────────────┐
  │ Citation normalization           │
  │  - renumber [Source N] markers   │
  │  - build SourceDocument list     │
  └────────┬─────────────────────────┘
           │
           ▼
  SSE "done" event → AgentResponse
  Session turn saved (fire-and-forget)
```

---

## 3. Entity & Temporal Resolution

**`_resolve_entities_via_llm()`** in `app/agents/simple_rag.py`

The resolver extracts structured `(ticker, call_date)` pairs from free-form queries like:
- "What did Salesforce say about AI last quarter?"
- "Compare NVDA and AMD guidance for Q3 FY2026"
- "Show me revenue across my portfolio for 2024"

**How it works:**

1. A gpt-4.1-mini call (JSON mode) receives the query + last 5 conversation turns and known company list from `_COMPANIES_CACHE`.
2. Output is a list of `{ticker, period}` intents where `period` can be a fiscal quarter label, relative phrase, or "latest".
3. `_batch_resolve_call_dates()` maps each intent to an actual `call_date` row from DB by querying available periods per ticker.
4. Rolling phrases like "last 2 quarters" or "FY2025" expand to multiple `(ticker, date)` pairs.
5. Results are keyed in `_SCOPE_CACHE` by `(normalized_query, companies_fingerprint, context_fingerprint)` — 24h TTL.

**Fiscal label → calendar date mapping** uses `app/rag/fiscal_calendar.py` (see §6).

**Multi-entity queries** return multiple pairs, e.g. a comparison of 3 companies × 2 quarters = 6 pairs, each retrieved independently and merged.

---

## 4. Retrieval Engine

**`app/rag/retriever.py`** → `retrieve_relevant_chunks()`

### Search Modes

| Mode | Mechanism | Index |
|------|-----------|-------|
| `vector` | pgvector cosine similarity (`<=>` operator) | IVFFlat (lists=100) |
| `keyword` | PostgreSQL full-text (`websearch_to_tsquery`) | GIN on `content_tsv` |
| `hybrid` | RRF merge of vector + keyword results | Both |

Default mode: `hybrid`.

### Reciprocal Rank Fusion

```python
rrf_score(doc) = sum(1 / (K + rank_in_mode))  # K=60
```

Both ranked lists are merged by `chunk_id`. Chunks appearing in both lists score higher than those in only one.

### Metadata Filtering

Retrieval narrows scope using dedicated B-tree columns (not JSONB — the legacy GIN index on `metadata` was dropped):

```sql
WHERE company_ticker = 'NVDA'
  AND call_date = '2025-08-28'
```

Three filter levels, most-to-least specific:
1. `_ticker_date_pairs` — exact `(ticker, date)` pairs from resolution
2. `_tickers` — company-only filter (all calls)
3. Standard field filters — arbitrary metadata key/value

### Financial Summary Injection

After chunk selection, `get_financials_chunks_for_pairs()` fetches a pre-computed financial summary chunk for each `(ticker, date)` pair (tagged `chunk_index==-1` or `data_source=="financial_summary"`) and prepends it to the context. This ensures key metrics (revenue, EPS, guidance) are always present even if not top-ranked by similarity.

### Per-Pair Chunk Budget

After RRF, chunks are grouped by `(ticker, call_date)` and trimmed to 3 chunks per pair. This prevents one highly-covered call from monopolizing context when a query spans multiple companies or periods.

---

## 5. Ingestion Pipeline

**`app/rag/ingestion.py`** → `ingest_document()`

### Speaker-Aware Chunking

Transcripts are split along speaker turns rather than fixed token windows. The chunker recognizes these header patterns:

```
"Name – Role: content..."     → executive/analyst format
"Name: content..."            → simple format
"OPERATOR:" / "MODERATOR:"   → standalone roles
```

Mid-sentence speaker insertions (nested headers) are handled. Large single-speaker blocks are sub-chunked at `chunk_size` (512 tokens) with `chunk_overlap` (64 tokens) overlap.

Each chunk carries metadata:
```json
{
  "speaker": "Jensen Huang",
  "role": "CEO",
  "company_ticker": "NVDA",
  "call_date": "2025-08-27",
  "fiscal_quarter": "Q2 FY2026",
  "period_end": "2025-07-27",
  "source_doc_id": "a1b2c3d4",
  "chunk_index": 3,
  "data_source": "earnings_transcript"
}
```

The first chunk (`chunk_index == 0`) also stores `_full_transcript` in metadata for full-text display in the UI.

### Document ID

```python
doc_id = sha256(f"{ticker}:{call_date}".encode()).hexdigest()[:16]
```

Re-ingesting the same `(ticker, call_date)` replaces existing chunks (upsert by `source_doc_id`).

### Financial Summary Extraction

After chunking, a separate LLM call extracts a structured financial summary (revenue, EPS, guidance, margins, YoY growth) and stores it as a special chunk. This chunk is always injected into context for relevant queries.

### Embedding

Embeddings generated via `text-embedding-3-small` (1536 dims), batched in groups of 100. Results cached in `_EMBEDDING_CACHE` (24h TTL).

---

## 6. Fiscal Calendar System

**`app/rag/fiscal_calendar.py`**

49 tickers have explicit fiscal year end month mappings. When no entry exists, December (calendar year) is assumed.

Examples:
| Ticker | FY End Month | Notes |
|--------|-------------|-------|
| MSFT | June | Q4 FY2025 = Apr–Jun 2025 |
| NVDA | January | Q3 FY2026 = Aug–Oct 2025 |
| PANW | July | Q1 FY2026 = Aug–Oct 2025 |
| ADBE | November | Q4 FY2025 = Sep–Nov 2025 |
| SNOW | January | same as NVDA |
| Most others | December | calendar year |

### Key Functions

```python
parse_fiscal_quarter("Q3 FY2026", ticker="NVDA")
# → FiscalQuarter(quarter=3, fiscal_year=2026)

compute_period_end(fq, ticker="NVDA")
# → date(2025, 10, 31)  # end of Q3 FY2026 for NVDA

cy_quarter_label_from_period_end(date(2025, 10, 31))
# → "CY Q3 2025"

period_end_to_label(date(2025, 10, 31))
# → "Aug–Oct 2025"
```

### Prompt Annotation Rule

Every fiscal quarter reference in answers must include its calendar period:

```
"Q3 FY2026 (Aug–Oct 2025)"
```

This lets users compare across companies with different fiscal calendars without mental conversion.

---

## 7. Streaming & SSE Protocol

**`app/agents/streaming.py`**

The `/agent/query` endpoint returns a `StreamingResponse` with `media_type="text/event-stream"`.

### Event Format

```
data: {"type": "delta", "content": "Revenue"}

data: {"type": "delta", "content": " grew"}

data: {"type": "done", "answer": "...", "sources": [...], "session_id": "..."}
```

- `delta` events carry incremental token strings.
- `done` carries the full `AgentResponse` payload including all source documents and citation spans.
- Frontend accumulates deltas for live rendering and processes `done` to attach source panels.

### Error Events

```
data: {"type": "error", "message": "Rate limit exceeded. Check your OpenAI quota."}
```

OpenAI `RateLimitError` is caught and classified: quota exhaustion vs. per-minute rate limiting, with distinct user-facing messages.

---

## 8. Conversation & Session Management

**`app/conversations/service.py`**

### Storage

`conversation_messages` table stores turns as:
```json
{
  "role": "user" | "assistant",
  "content": "<text>",
  "sources": [...],   // assistant turns only
  "title": "...",     // first user message used as session title
  "position": 0       // ordering within session
}
```

Sessions are keyed by `session_id` (UUID generated by frontend) and optionally associated with a `user_id` FK (if authenticated).

### In-Memory Buffer

`_SESSION_RECENT_QUERIES` (TTLCache, 2000 sessions, 2h TTL) stores a `deque` of the last 5 `(user_text, assistant_text)` pairs per session.

**Write order matters:** the buffer is written synchronously before the async DB write. This means the next query in the same session can use context immediately without waiting for the DB round-trip.

### Context Injection

`get_recent_turns(history, limit=5)` extracts the last N turns and formats them for the entity resolution prompt:

```
[Turn 1]
User: What was NVDA revenue last quarter?
Assistant: NVDA Q3 FY2026 revenue was $35.1B...

[Turn 2]
User: How does that compare to AMD?
```

The resolver sees the full prior context, enabling pronoun resolution ("that", "them") and carry-forward entity references.

---

## 9. Caching Architecture

All caches are **in-process only**. On multi-instance deployments (e.g., Render with 2 replicas), each process has an independent cache — no distributed invalidation.

| Cache | Type | Capacity | TTL | Key | Invalidation |
|-------|------|----------|-----|-----|-------------|
| `_EMBEDDING_CACHE` | TTLCache | 1000 | 24h | `(text, model, dims)` | Never (TTL only) |
| `_SCOPE_CACHE` | TTLCache | 500 | 24h | `(query, companies_fp, ctx_fp)` | Never (TTL only) |
| `_SESSION_RECENT_QUERIES` | TTLCache | 2000 | 2h | `session_id` | Never (TTL only) |
| `_COMPANIES_CACHE` | TTLCache | 1 | 24h | `"companies"` | On ingest |
| `_PERIODS_CACHE` | TTLCache | 1 | 24h | `"periods"` | On ingest |
| `_relevance_cache` | dict | unbounded | session | `hash(query+chunk)` | Process restart |

### Warmup

`GET /warmup` triggers:
1. `get_known_companies()` — populates `_COMPANIES_CACHE` via DB query
2. `get_openai_client()` — initializes the AsyncOpenAI singleton

The lifespan handler (`@app.on_event("startup")`) also calls warmup. The `/warmup` route is a fallback for when startup completes before the DB is ready (common on cold Render deploys). Frontend calls it on mount.

---

## 10. Auth System

**`app/auth/`**

### Flow

```
Frontend                     Backend                    Google
   │                            │                          │
   │── GET /auth/google ────────►│                          │
   │◄── 302 redirect ───────────│── oauth2 authorize ──────►│
   │                            │                          │
   │── GET /auth/google/callback?code=... ─────────────────►│
   │                            │◄── token exchange ────────│
   │                            │◄── user info (email) ─────│
   │                            │                          │
   │                            │── upsert_user() ─────────►│ DB
   │◄── redirect + JWT cookie ──│                          │
```

### JWT Structure

```json
{
  "sub": "<user_uuid>",
  "email": "user@example.com",
  "name": "User Name",
  "avatar_url": "https://...",
  "exp": 1234567890
}
```

Algorithm: HS256. Expiry: 30 days (`JWT_EXPIRE_DAYS`). Secret: `JWT_SECRET` env.

### FastAPI Dependencies

```python
# Requires valid JWT — raises 401 if missing/expired
user = Depends(require_user)

# Returns decoded payload or None
user = Depends(get_optional_user)
```

Agent queries use `get_optional_user` — unauthenticated queries are allowed but session turns won't be linked to a user account.

---

## 11. Eval Framework

**`app/evals/`**

### Isolation

Eval runs use `eval_document_chunks` (identical schema to `document_chunks`) so production data is never polluted. A `ContextVar` (`_USE_EVAL_TABLE`) switches the retrieval table:

```python
async with use_eval_chunks_context():
    result = await run_eval("dataset_name")
    # All retrieval hits eval_document_chunks
```

### Agent Evals

Dataset format (`eval_datasets/*.json`):
```json
{
  "cases": [
    {
      "query": "What was NVDA gross margin in Q3 FY2026?",
      "expected_answer": "NVDA gross margin was 74.6% ...",
      "expected_sources": ["NVDA_2025-08-28"],
      "tags": ["margin", "single_company"]
    }
  ]
}
```

Each case runs through the same `stream_simple_rag_or_agent()` pipeline. Three LLM judges score the response:

| Evaluator | Prompt | Score range |
|-----------|--------|-------------|
| `FaithfulnessEvaluator` | "Is the answer supported by the provided context?" | 0–1 |
| `RelevanceEvaluator` | "Does the answer address the question?" | 0–1 |
| `CompletenessEvaluator` | "Does the answer include all key facts from the reference?" | 0–1 |

### Retrieval Evals

`POST /evals/retrieval` compares vector/keyword/hybrid modes on ground-truth source lists:

```
For each eval case:
  For each mode in [vector, keyword, hybrid]:
    retrieved = retrieve_relevant_chunks(query, mode=mode)
    Compare retrieved doc titles vs expected_sources

Aggregate:
  precision@k   = |retrieved ∩ expected| / |retrieved|
  recall@k      = |retrieved ∩ expected| / |expected|
  MRR           = mean(1/rank_of_first_relevant_doc)
  hit@k         = 1 if any expected in top-k else 0
```

Relevance judgments are cached by `hash(query + chunk_content)` to avoid duplicate LLM calls when a chunk appears in multiple modes.

### Scripts

```bash
# Run agent evals, print overall scores + per-case results
python scripts/run_evals.py [dataset_name] --base-url https://api.example.com

# Run retrieval evals, print precision/recall/MRR per mode
python scripts/run_retrieval_evals.py [dataset_name] --base-url https://api.example.com
```

Both default to `http://localhost:8000` if `--base-url` is not specified.

---

## 12. Database Schema

```sql
-- Google OAuth users
CREATE TABLE users (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    google_id   TEXT UNIQUE NOT NULL,
    email       TEXT UNIQUE NOT NULL,
    name        TEXT,
    avatar_url  TEXT,
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);

-- Production RAG chunks
CREATE TABLE document_chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content         TEXT NOT NULL,
    embedding       VECTOR(1536),
    metadata        JSONB DEFAULT '{}',
    company_ticker  TEXT,           -- B-tree indexed
    call_date       DATE,           -- B-tree indexed
    fiscal_quarter  TEXT,           -- e.g. "Q3 FY2026"
    period_end      DATE,           -- calendar period end
    source_doc_id   TEXT,           -- sha256(ticker:date)[:16]
    chunk_index     INTEGER,
    content_tsv     TSVECTOR GENERATED ALWAYS AS (
                        to_tsvector('english', content)
                    ) STORED,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON document_chunks USING GIN (content_tsv);
CREATE INDEX ON document_chunks (company_ticker, call_date);

-- Identical schema — eval data only
CREATE TABLE eval_document_chunks ( ... );

-- Conversation history
CREATE TABLE conversation_messages (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  TEXT NOT NULL,
    user_id     UUID REFERENCES users(id),
    role        TEXT NOT NULL,      -- "user" | "assistant"
    content     JSONB NOT NULL,     -- {text: "..."}
    sources     JSONB,              -- SourceDocument list (assistant turns)
    title       TEXT,               -- session title (first user message)
    position    INTEGER NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX ON conversation_messages (session_id);

-- Eval run results
CREATE TABLE eval_results (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id       TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    scores       JSONB NOT NULL,
    details      JSONB DEFAULT '{}',
    created_at   TIMESTAMPTZ DEFAULT now()
);

-- Prompt versioning (A/B testing support)
CREATE TABLE prompt_versions (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name       TEXT NOT NULL,
    version    INTEGER NOT NULL,
    template   TEXT NOT NULL,
    is_active  BOOLEAN DEFAULT false,
    weight     FLOAT DEFAULT 0.0,
    metadata   JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(name, version)
);
```

---

## 13. Configuration Reference

All settings in `app/config.py`, loaded from environment variables via Pydantic `BaseSettings`.

| Variable | Default | Notes |
|----------|---------|-------|
| `DATABASE_URL` | required | `postgresql+asyncpg://...` |
| `OPENAI_API_KEY` | required | |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | |
| `EMBEDDING_DIMENSIONS` | `1536` | |
| `CHUNK_SIZE` | `512` | tokens |
| `CHUNK_OVERLAP` | `64` | tokens |
| `RETRIEVAL_TOP_K` | `20` | candidates per mode before RRF |
| `RETRIEVAL_THRESHOLD` | `0.3` | minimum cosine similarity |
| `DEFAULT_SEARCH_MODE` | `hybrid` | vector / keyword / hybrid |
| `CONTEXT_TOKEN_BUDGET` | `20000` | max tokens in system prompt context |
| `AGENT_TEMPERATURE` | `0.1` | LLM temperature |
| `MAX_TOOL_CALLS` | `5` | |
| `EVAL_CONCURRENCY` | `3` | parallel eval cases |
| `ADMIN_API_KEY` | `""` | empty = no protection |
| `CORS_ORIGINS` | `["http://localhost:5173"]` | comma-separated in env |
| `LOG_LEVEL` | `INFO` | |
| `GOOGLE_CLIENT_ID` | `""` | optional — OAuth disabled if empty |
| `GOOGLE_CLIENT_SECRET` | `""` | |
| `JWT_SECRET` | required if auth enabled | HS256 signing key |
| `JWT_ALGORITHM` | `HS256` | |
| `JWT_EXPIRE_DAYS` | `30` | |
| `FRONTEND_URL` | `http://localhost:5173` | OAuth redirect base |
| `BACKEND_URL` | `http://localhost:8000` | OAuth callback base |

> `OPENAI_MODEL` env is **ignored**. The model is hardcoded to `gpt-4.1-mini` in `app/config.py`.
