"""Shared Pydantic schemas for request/response models."""

from datetime import date, datetime, timezone
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, PrivateAttr


# --- Agent Schemas ---


SearchModeType = Literal["vector", "keyword", "hybrid"]


class QueryRequest(BaseModel):
    """Request to query the KB agent.

    Only ``query`` is required. All other fields fall back to values
    defined in ``Settings`` (see ``app/config.py``) when omitted.
    """
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = Field(
        default=None,
        description="Session ID for multi-turn conversations. Omit to start a new session.",
    )
    search_mode: SearchModeType | None = Field(
        default=None,
        description="Retrieval mode: vector, keyword, or hybrid. Falls back to DEFAULT_SEARCH_MODE in config.",
    )
    retrieval_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Min similarity for vector search (0–1). Falls back to RETRIEVAL_THRESHOLD in config.",
    )


class CitedSpan(BaseModel):
    """Character range in source content that was cited in the answer."""
    start: int = Field(..., ge=0, description="Start index (inclusive)")
    end: int = Field(..., ge=0, description="End index (exclusive)")


class SourceDocument(BaseModel):
    """Retrieved chunk: used as source in agent responses and RAG semantic search results."""
    chunk_id: str
    content: str
    similarity: float
    metadata: dict[str, Any] = {}
    cited_spans: list[CitedSpan] = Field(
        default_factory=list,
        description="Character ranges in content that were cited in the answer (for highlighting).",
    )


class Citation(BaseModel):
    """A quote from context used in the answer. source_index is 1-based (Source 1, Source 2, ...)."""
    source_index: int = Field(..., ge=1, description="1-based index of the source in the context")
    quote: str = Field(..., min_length=1, description="Exact or close quote from that source used in the answer")


class AgentResponse(BaseModel):
    """Structured response from the KB agent."""
    answer: str = Field(..., description="The agent's answer to the query")
    confidence: float = Field(
        ...,
        description="Confidence score from 0.0 to 1.0",
        ge=0.0,
        le=1.0,
    )
    sources: list[SourceDocument] = []
    citations: list[Citation] = Field(
        default_factory=list,
        description="Quotes from the context used in the answer. source_index is 1-based (Source 1 = 1).",
    )
    reasoning: str | None = Field(None, description="Chain-of-thought reasoning")
    tool_calls_made: list[str] = []
    _raw_retrieved_chunks: list[dict[str, Any]] = PrivateAttr(default_factory=list)


# --- RAG Schemas ---


class IngestResult(BaseModel):
    """Result of document ingestion."""
    doc_id: str
    title: str
    chunks_created: int
    status: str = "success"


class ManualIngestMissingResponse(BaseModel):
    """Returned when manual ingest is called without required fields."""
    missing: list[str] = Field(
        ...,
        description="List of required field names that were not provided.",
    )
    message: str = Field(
        ...,
        description="Human-readable instructions for the user.",
    )


class SearchRequest(BaseModel):
    """Direct semantic search request."""
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    company_ticker: str | None = Field(
        None,
        description="Scope results to a specific company.",
    )
    search_mode: SearchModeType = Field(
        default="hybrid",
        description="Retrieval mode: vector (embedding), keyword (BM25/full-text), hybrid (RRF merge)",
    )


# --- Eval Schemas ---


class EvalCase(BaseModel):
    """A single evaluation test case."""
    query: str
    expected_answer: str | None = None
    expected_sources: list[str] = []
    tags: list[str] = []


class EvalDataset(BaseModel):
    """A dataset of evaluation cases."""
    name: str
    description: str = ""
    cases: list[EvalCase]


class MetricScore(BaseModel):
    """Score for a single metric."""
    name: str
    score: float = Field(..., ge=0.0, le=1.0)
    details: str = ""


class RetrievedChunk(BaseModel):
    """Serializable snapshot of a retrieved chunk for eval output."""
    chunk_id: str
    content: str
    similarity: float
    metadata: dict[str, Any] = {}


class RetrievalMetrics(BaseModel):
    """Per-case retrieval quality metrics."""
    num_chunks_retrieved: int = 0
    precision_at_k: float = Field(0.0, ge=0.0, le=1.0)
    recall_at_k: float = Field(0.0, ge=0.0, le=1.0)
    mrr: float = Field(0.0, ge=0.0, le=1.0)
    hit_at_k: float = Field(0.0, ge=0.0, le=1.0)


class EvalCaseResult(BaseModel):
    """Result of evaluating a single case."""
    query: str
    response: str
    scores: list[MetricScore]
    latency_ms: float
    retrieved_chunks: list[RetrievedChunk] = []
    retrieval_metrics: RetrievalMetrics | None = None


class EvalRunResult(BaseModel):
    """Result of a full eval run."""
    run_id: str
    dataset_name: str
    overall_scores: dict[str, float]
    overall_retrieval_metrics: RetrievalMetrics | None = None
    case_results: list[EvalCaseResult]
    total_latency_ms: float
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
