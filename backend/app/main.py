"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load .env from backend directory
_backend = Path(__file__).resolve().parent.parent
load_dotenv(_backend / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from app.agents.router import router as agent_router
from app.auth.router import router as auth_router
from app.config import get_settings
from app.conversations.router import router as conversations_router
from app.evals.router import router as evals_router
from app.models.database import health_check
from app.rag.router import router as rag_router

# Configure logging
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("KB Agent starting up...")
    logger.info(f"Model: {settings.openai_model}")
    logger.info(f"Embedding model: {settings.embedding_model}")
    yield
    logger.info("KB Agent shutting down...")


# Create app
app = FastAPI(
    title="KB Agent",
    description="Production knowledge base agent with PydanticAI, RAG, and evals",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — no allow_credentials since session IDs are passed in request body, not cookies
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth_router)
app.include_router(agent_router)
app.include_router(conversations_router)
app.include_router(rag_router)
app.include_router(evals_router)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    schema.setdefault("components", {})["securitySchemes"] = {
        "AdminKey": {"type": "apiKey", "in": "header", "name": "X-Admin-Key"}
    }
    schema["security"] = [{"AdminKey": []}]
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi


@app.get("/health")
async def health():
    """Health check endpoint."""
    db_health = await health_check()
    return {
        "status": "ok",
        "service": "kb-agent",
        "version": "0.1.0",
        "database": db_health,
    }
