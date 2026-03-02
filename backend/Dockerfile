# Earnings Analyzer Backend — Docker image for FastAPI
FROM python:3.12-slim

WORKDIR /app

# Install system deps (if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY alembic/ ./alembic/
COPY alembic.ini .
COPY eval_data/ ./eval_data/
COPY eval_datasets/ ./eval_datasets/
COPY scripts/ ./scripts/

# Expose default port (Railway/Cloud Run use PORT env)
ENV PORT=8000
EXPOSE 8000

# Run migrations, then uvicorn (PORT overridden by Railway/Cloud Run)
CMD alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
