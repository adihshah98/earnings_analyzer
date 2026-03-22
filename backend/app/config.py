"""Application configuration via pydantic-settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI (gpt-4.1-nano is fastest; set OPENAI_MODEL=gpt-4o-mini for higher quality)
    openai_api_key: str
    openai_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Database (PostgreSQL via asyncpg)
    database_url: str  # e.g. postgresql+asyncpg://user:pass@host:5432/dbname

    # RAG
    chunk_size: int = 512
    chunk_overlap: int = 64
    retrieval_top_k: int = 20
    retrieval_threshold: float = 0.3
    default_search_mode: str = "hybrid"
    context_token_budget: int = 80_000  # max tokens for retrieved context in the system prompt

    # Agent
    max_tool_calls: int = 5
    agent_temperature: float = 0.1
    default_company_ticker: str | None = None

    # Evals
    eval_concurrency: int = 3

    # Auth (Google OAuth + JWT)
    google_client_id: str = ""
    google_client_secret: str = ""
    jwt_secret: str = "dev-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_days: int = 30
    frontend_url: str = "http://localhost:5173"
    backend_url: str = "http://localhost:8000"  # used as OAuth redirect_uri base

    # Server
    log_level: str = "INFO"
    admin_api_key: str = ""  # empty = admin routes unprotected (dev only)
    cors_origins: list[str] = ["http://localhost:5173"]


@lru_cache()
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
