"""Tools available to the PydanticAI KB agent."""

import logging
import math
from typing import Any

from pydantic_ai import RunContext

from app.conversations.service import get_conversation_history, get_recent_user_queries
from app.rag.retriever import (
    list_available_transcripts as _list_available_transcripts,
    retrieve_relevant_chunks,
)

logger = logging.getLogger(__name__)


# Type alias for the agent's dependency context
class AgentDeps:
    """Dependencies injected into the agent at runtime."""

    def __init__(
        self,
        session_id: str | None = None,
        user_name: str = "anonymous",
        user_role: str = "viewer",
        user_department: str = "general",
        custom_instructions: str = "",
        metadata: dict[str, Any] | None = None,
        search_mode: str | None = None,
        retrieval_threshold: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
        today_date: str = "",
    ):
        self.session_id = session_id
        self.user_name = user_name
        self.user_role = user_role
        self.user_department = user_department
        self.custom_instructions = custom_instructions
        self.metadata = metadata or {}
        self.search_mode = search_mode
        self.retrieval_threshold = retrieval_threshold
        self.filter_metadata = filter_metadata or {}
        self.today_date = today_date
        self.retrieved_chunks: list[dict[str, Any]] = []


async def search_knowledge_base(
    ctx: RunContext[AgentDeps],
    query: str,
    top_k: int = 5,
    search_mode: str = "hybrid",
    company_ticker: str | None = None,
    as_of_date: str | None = None,
) -> str:
    """Search the knowledge base for relevant earnings call information.

    Args:
        ctx: The agent run context with dependencies.
        query: The search query.
        top_k: Number of results to return.
        search_mode: Retrieval mode: "vector" (semantic), "keyword" (BM25/full-text),
            or "hybrid" (combines both via RRF). Default from request or "hybrid".
        company_ticker: Company ticker to filter results (e.g. ACME, GLBX, INIT).
            ALWAYS set this when the user's question mentions a specific company.
            This prevents cross-company contamination in search results.
        as_of_date: Reference date for temporal scoping (ISO format). Omit for "latest"
            — the system uses today's date automatically. Otherwise use a call_date
            from list_available_transcripts or a date at end of period (e.g. "Q3 2024" → "2024-12-31").

    Returns:
        Formatted string of relevant document chunks.
    """
    # Per-request overrides from API (deps) take precedence over tool arg default
    effective_mode = ctx.deps.search_mode if ctx.deps.search_mode is not None else search_mode
    threshold = ctx.deps.retrieval_threshold

    # Build filter: API filters from deps, tool params from agent (extracted from query)
    # Use server-provided today_date when agent omits as_of_date (avoids LLM inventing wrong dates)
    filter_metadata = dict(ctx.deps.filter_metadata)
    if company_ticker:
        filter_metadata["company_ticker"] = company_ticker
    effective_as_of = as_of_date or (ctx.deps.today_date if ctx.deps.today_date else None)
    if effective_as_of:
        filter_metadata["as_of_date"] = effective_as_of
    filter_metadata = filter_metadata or None

    logger.info(f"[Tool:search_kb] query='{query}', top_k={top_k}, search_mode={effective_mode}, filters={filter_metadata}")

    # Use conversation context for better retrieval in multi-turn conversations
    conversation_context: list[str] | None = None
    if ctx.deps.session_id:
        history = await get_conversation_history(ctx.deps.session_id)
        recent = get_recent_user_queries(history, limit=3)
        if recent:
            conversation_context = recent

    chunks = await retrieve_relevant_chunks(
        query=query,
        top_k=top_k,
        threshold=threshold,
        filter_metadata=filter_metadata,
        conversation_context=conversation_context,
        search_mode=effective_mode,
    )

    if not chunks:
        return "No relevant documents found in the knowledge base."

    ctx.deps.retrieved_chunks.extend(chunks)

    formatted_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("title", "Unknown")
        similarity = chunk.get("similarity", 0.0)
        content = chunk["content"]
        formatted_parts.append(
            f"[Source {i}: {source} (relevance: {similarity:.2f})]\n{content}"
        )

    return "\n\n---\n\n".join(formatted_parts)


async def list_available_transcripts(
    ctx: RunContext[AgentDeps],
    company_ticker: str | None = None,
) -> str:
    """List available earnings call transcripts in the knowledge base.

    ALWAYS call this FIRST when the user asks about a specific company, before
    calling search_knowledge_base. Use the returned call dates to choose a valid
    as_of_date for search. Also call when the user asks what data is available.

    Args:
        ctx: The agent run context.
        company_ticker: Optional filter to list only transcripts for one company.

    Returns:
        Formatted list of (company, call_date, title) for available transcripts.
    """
    logger.info(f"[Tool:list_transcripts] company_ticker={company_ticker}")
    transcripts = await _list_available_transcripts(company_ticker=company_ticker)
    if not transcripts:
        return "No earnings call transcripts found in the knowledge base."
    lines = [
        f"- {t['company_ticker']} | {t['call_date']} | {t['title']}"
        for t in transcripts
    ]
    return "Available transcripts:\n" + "\n".join(lines)


async def calculate(
    ctx: RunContext[AgentDeps],
    expression: str,
) -> str:
    """Evaluate a mathematical expression.

    Args:
        ctx: The agent run context.
        expression: A mathematical expression to evaluate (e.g., "2 + 2", "sqrt(16)").

    Returns:
        The result as a string, or an error message if evaluation fails.
    """
    logger.info(f"[Tool:calculate] expression='{expression}'")

    safe_globals = {
        "__builtins__": {},
        "math": math,
        "sqrt": math.sqrt,
        "abs": abs,
        "round": round,
        "pow": pow,
        "min": min,
        "max": max,
    }

    try:
        result = eval(expression, safe_globals)
        return str(result)
    except Exception as e:
        return f"Error: could not evaluate expression: {e!s}"


