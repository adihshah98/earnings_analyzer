# Backend Deployment (Docker)

Steps to build and run the backend with Docker, and deploy to Railway or similar platforms.

## 1. Build the image

From the **backend** directory:

```bash
cd backend
docker build -t earnings-analyzer-backend .
```

## 2. Run locally

You need a PostgreSQL database with pgvector. Set environment variables (or use a `.env` file):

```bash
docker run -p 8000:8000 \
  -e DATABASE_URL="postgresql+asyncpg://user:password@host.docker.internal:5432/dbname" \
  -e OPENAI_API_KEY="sk-..." \
  -e CORS_ORIGINS="http://localhost:5173" \
  earnings-analyzer-backend
```

For a local Postgres on the host, use `host.docker.internal` (Mac/Windows) or your machine’s IP (Linux).

## 3. Migrations

Migrations run automatically on container startup (`alembic upgrade head` before uvicorn). No separate step required.

## 4. Deploy to Railway

1. Create a project and add **PostgreSQL with pgvector** (not the standard Postgres template).
2. Add a new service → **Deploy from GitHub** → select your repo.
3. Configure:
   - **Root Directory:** `backend`
   - **Build:** Railway will detect the Dockerfile and build the image.
   - **Variables:** Add at least:
     - `DATABASE_URL` — use Railway’s Postgres reference, but change `postgresql://` to `postgresql+asyncpg://`
     - `OPENAI_API_KEY`
     - `CORS_ORIGINS` — your frontend URL(s)
4. Generate a public domain in **Settings → Networking**.
5. Migrations run on each deploy; no separate step needed.

## 5. Environment variables

| Variable | Required | Notes |
|----------|----------|-------|
| `DATABASE_URL` | Yes | `postgresql+asyncpg://user:pass@host:5432/dbname` |
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `EMBEDDING_MODEL` | No | Default: `text-embedding-3-small` |
| `EMBEDDING_DIMENSIONS` | No | Default: `1536` |
| `CORS_ORIGINS` | No | Comma-separated origins; default: `http://localhost:5173` |
| `ADMIN_API_KEY` | No | For ingest/eval routes; empty = unprotected (dev default) |
| `LOG_LEVEL` | No | Default: `INFO` |
| `PORT` | No | Port to bind; set by Railway/Cloud Run |
| `GOOGLE_CLIENT_ID` | No | Google OAuth app client ID; auth disabled if empty |
| `GOOGLE_CLIENT_SECRET` | No | Google OAuth app client secret |
| `JWT_SECRET` | If auth enabled | HS256 signing key for session JWTs |
| `JWT_EXPIRE_DAYS` | No | Default: `30` |
| `FRONTEND_URL` | If auth enabled | OAuth redirect base (e.g. `https://your-frontend.com`) |
| `BACKEND_URL` | If auth enabled | OAuth callback base (e.g. `https://your-api.com`) |

Chat completions always use **`gpt-4.1-mini`** (`DEFAULT_OPENAI_MODEL` in `app/config.py`). The `OPENAI_MODEL` environment variable is ignored.
