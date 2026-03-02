# 🧠 KB Agent — Production Knowledge Base Agent

A production-grade FastAPI service that provides an AI-powered knowledge base agent with:
- **PydanticAI agents** with tool use (search, calculate, summarize)
- **RAG pipeline** using Supabase pgvector for semantic search
- **Prompt versioning** with A/B testing and rollback
- **Structured output validation** via Pydantic models
- **Eval harness** for systematic agent quality testing

> For detailed technical documentation (architecture, workflows, components), see [ARCHITECTURE.md](ARCHITECTURE.md).

## Architecture

```
Earnings Analyzer/
├── backend/
│   ├── app/                    # FastAPI application
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── agents/, rag/, evals/, models/, tools/, conversations/, prompts/
│   │   └── ingestion/
│   ├── scripts/
│   │   ├── seed_transcript.py  # Seed transcripts from eval_data/*.json (local)
│   │   ├── run_evals.py        # Agent evals via POST /evals/run
│   │   ├── run_retrieval_evals.py  # Retrieval evals via POST /evals/retrieval
│   │   ├── smoke_test.py
│   │   └── show_chunks.py
│   ├── eval_data/              # Transcript JSON files for seeding
│   ├── eval_datasets/          # Eval case definitions
│   ├── tests/
│   ├── alembic/
│   ├── pyproject.toml
│   ├── requirements.txt
│   ├── alembic.ini
│   └── .env.example
├── frontend/                   # React + Vite
└── README.md
```

## Setup

### Prerequisites
- Python 3.11+
- OpenAI API key
- PostgreSQL with pgvector (Supabase, Render Postgres, or self-hosted)

### Database Setup

**Option A: Use Alembic migrations (recommended)**

```bash
cd backend && alembic upgrade head
```

**Option B: Run SQL manually**

If you prefer not to use Alembic, run this SQL in your PostgreSQL database (Supabase SQL editor, Render Postgres console, or psql):

```sql
-- Enable pgvector
create extension if not exists vector;

-- Document chunks for RAG
create table document_chunks (
    id uuid default gen_random_uuid() primary key,
    content text not null,
    metadata jsonb default '{}',
    embedding vector(1536),
    source_doc_id text,
    chunk_index integer,
    created_at timestamptz default now()
);

-- Index for similarity search
create index on document_chunks
using ivfflat (embedding vector_cosine_ops)
with (lists = 100);

-- Prompt versions
create table prompt_versions (
    id uuid default gen_random_uuid() primary key,
    name text not null,
    version integer not null,
    template text not null,
    is_active boolean default false,
    weight float default 0.0,
    metadata jsonb default '{}',
    created_at timestamptz default now(),
    unique(name, version)
);

-- Eval results
create table eval_results (
    id uuid default gen_random_uuid() primary key,
    run_id text not null,
    dataset_name text not null,
    prompt_version_id uuid references prompt_versions(id),
    scores jsonb not null,
    details jsonb default '{}',
    created_at timestamptz default now()
);

-- Match function for similarity search
create or replace function match_documents(
    query_embedding vector(1536),
    match_threshold float,
    match_count int
)
returns table (
    id uuid,
    content text,
    metadata jsonb,
    similarity float
)
language sql stable
as $$
    select
        document_chunks.id,
        document_chunks.content,
        document_chunks.metadata,
        1 - (document_chunks.embedding <=> query_embedding) as similarity
    from document_chunks
    where 1 - (document_chunks.embedding <=> query_embedding) > match_threshold
    order by (document_chunks.embedding <=> query_embedding)
    limit match_count;
$$;
```

### Install & Run

```bash
# Clone and setup
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r backend/requirements.txt

# Configure backend
cp backend/.env.example backend/.env
# Edit backend/.env: OPENAI_API_KEY, DATABASE_URL (postgresql+asyncpg://...)

# Run migrations
cd backend && alembic upgrade head

# Seed transcripts (from backend/)
cd backend && python scripts/seed_transcript.py
# Or with --eval to ingest into eval_document_chunks

# Run the API server (from backend/)
cd backend && uvicorn app.main:app --reload --port 8000

# Run evals (calls the deployed API — start server first)
cd backend && python scripts/run_evals.py [dataset]
cd backend && python scripts/run_retrieval_evals.py [dataset]
# Use --base-url https://api.example.com for remote; defaults to http://localhost:8000
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/agent/query` | Query the KB agent |
| GET | `/conversations/{session_id}/history` | Get conversation history for a session |
| POST | `/rag/ingest` | Ingest documents |
| POST | `/rag/search` | Direct semantic search (supports search_mode: vector, keyword, hybrid) |
| GET | `/prompts/` | List all prompt versions |
| POST | `/prompts/` | Create new prompt version |
| PUT | `/prompts/{name}/activate` | Activate a prompt version |
| POST | `/evals/run` | Run eval suite |
| POST | `/evals/compare` | Compare two prompt versions (bootstrap CI) |
| POST | `/evals/retrieval` | Compare retrieval quality across vector/keyword/hybrid modes |
| GET | `/evals/comparisons` | Get stored comparison results |
| GET | `/evals/dashboard` | HTML dashboard for prompt comparisons |
| GET | `/evals/results` | Get eval results |
| GET | `/health` | Health check |

## Docker deployment

To deploy the backend with Docker (e.g. Railway), see [backend/DEPLOY.md](backend/DEPLOY.md). Summary:

```bash
cd backend
docker build -t earnings-analyzer-backend .
# Set DATABASE_URL, OPENAI_API_KEY, CORS_ORIGINS
docker run -p 8000:8000 -e DATABASE_URL=... -e OPENAI_API_KEY=... earnings-analyzer-backend
```

Migrations run on container startup automatically.

## Tech Stack
- **Framework:** FastAPI + Pydantic
- **Agent:** PydanticAI (with OpenAI)
- **Embeddings:** OpenAI text-embedding-3-small
- **Database:** PostgreSQL via asyncpg + SQLAlchemy + pgvector
- **Eval:** Pydantic Evals
