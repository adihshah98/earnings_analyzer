"""Main PydanticAI knowledge base agent."""

import logging
from datetime import date
from typing import Any

from pydantic_ai import Agent, UsageLimits

import asyncio

from app.config import get_settings
from app.conversations.service import (
    append_conversation_messages,
    get_conversation_history,
    get_recent_user_queries,
)
from app.models.schemas import AgentResponse, Citation, CitedSpan, SourceDocument
from app.prompts.templates import SYSTEM_PROMPT_V1
from app.rag.retriever import get_known_companies, retrieve_relevant_chunks
from app.tools.agent_tools import (
    AgentDeps,
    calculate,
    list_available_transcripts,
    search_knowledge_base,
)

logger = logging.getLogger(__name__)


def _find_quote_spans(content: str, quote: str) -> list[tuple[int, int]]:
    """Find all occurrences of quote (exact, stripped) in content. Returns list of (start, end)."""
    q = quote.strip()
    if not q:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        pos = content.find(q, start)
        if pos == -1:
            break
        spans.append((pos, pos + len(q)))
        start = pos + 1
    return spans


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent spans. Input must be (start, end) with end exclusive."""
    if not spans:
        return []
    sorted_spans = sorted(spans, key=lambda s: s[0])
    merged = [sorted_spans[0]]
    for start, end in sorted_spans[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _apply_citations_to_sources(
    sources: list[SourceDocument],
    citations: list[Citation],
) -> None:
    """Fill cited_spans on each source from the model's citations. Mutates sources in place."""
    for i, source in enumerate(sources):
        source_1based = i + 1
        all_spans: list[tuple[int, int]] = []
        for c in citations:
            if c.source_index != source_1based:
                continue
            all_spans.extend(_find_quote_spans(source.content, c.quote))
        source.cited_spans = [
            CitedSpan(start=s, end=e) for s, e in _merge_spans(all_spans)
        ]


def _sanitize_for_prompt(value: str, max_len: int = 200) -> str:
    """Sanitize user-controlled values before interpolating into prompts.

    Prevents prompt injection by normalizing newlines/control chars and truncating.
    """
    if not value:
        return value
    sanitized = " ".join(str(value).split())
    return sanitized[:max_len] if len(sanitized) > max_len else sanitized


def _format_known_tickers(companies: list[dict[str, str]]) -> str:
    """Format a list of company dicts into a prompt-friendly string.

    >>> _format_known_tickers([{"ticker": "ACME", "name": "Acme Corp"}])
    'ACME (Acme Corp)'
    """
    if not companies:
        return "none — use list_available_transcripts to discover companies"
    return ", ".join(f"{c['ticker']} ({c['name']})" for c in companies)


def _build_system_prompt(
    template: str,
    deps: AgentDeps,
    context: str = "",
    known_tickers: str = "",
) -> str:
    """Build the system prompt from a template and dependencies."""
    user_name = _sanitize_for_prompt(deps.user_name)
    user_role = _sanitize_for_prompt(deps.user_role)
    user_department = _sanitize_for_prompt(deps.user_department)
    custom_instructions = _sanitize_for_prompt(deps.custom_instructions, max_len=500)
    today_date = deps.today_date or "unknown"

    try:
        return template.format(
            context=context,
            user_name=user_name,
            user_role=user_role,
            user_department=user_department,
            custom_instructions=custom_instructions,
            known_tickers=known_tickers,
            today_date=today_date,
        )
    except KeyError:
        return (
            template.replace("{context}", context)
            .replace("{known_tickers}", known_tickers)
            .replace("{today_date}", today_date)
        )


def create_kb_agent() -> Agent[AgentDeps, AgentResponse]:
    """Create and configure the PydanticAI knowledge base agent.

    Returns:
        Configured PydanticAI agent with tools registered.
    """
    settings = get_settings()

    agent = Agent(
        model=f"openai:{settings.openai_model}",
        output_type=AgentResponse,
        system_prompt=SYSTEM_PROMPT_V1,  # Default; overridden at runtime
        deps_type=AgentDeps,
        retries=2,
    )

    # Register tools
    agent.tool(search_knowledge_base)
    agent.tool(list_available_transcripts)
    agent.tool(calculate)

    return agent


# Global agent instance
kb_agent = create_kb_agent()


async def query_agent(
    query: str,
    session_id: str | None = None,
    user_name: str = "anonymous",
    user_role: str = "viewer",
    user_department: str = "general",
    custom_instructions: str = "",
    metadata: dict[str, Any] | None = None,
    search_mode: str | None = None,
    retrieval_threshold: float | None = None,
    filter_metadata: dict[str, Any] | None = None,
) -> AgentResponse:
    """Run a query through the KB agent.

    Args:
        query: The user's question.
        session_id: Optional session ID for conversation tracking.
        user_name: Name of the requesting user.
        user_role: Role/access level of the user.
        user_department: Department of the user.
        custom_instructions: Additional instructions to inject.
        metadata: Additional metadata.
        search_mode: Retrieval mode (vector, keyword, hybrid). None uses tool default.
        retrieval_threshold: Min similarity for vector search (0–1). None uses app config.
        filter_metadata: Metadata filters for RAG retrieval (e.g. company_ticker).

    Returns:
        Structured AgentResponse with answer, sources, confidence, etc.
    """

    deps = AgentDeps(
        session_id=session_id,
        user_name=user_name,
        user_role=user_role,
        user_department=user_department,
        custom_instructions=custom_instructions,
        metadata=metadata or {},
        search_mode=search_mode,
        retrieval_threshold=retrieval_threshold,
        filter_metadata=filter_metadata or {},
        today_date=date.today().isoformat(),
    )

    system_prompt = SYSTEM_PROMPT_V1

    # Dynamically resolve known tickers from the DB so the prompt stays
    # in sync with whatever transcripts have been ingested.
    try:
        companies = await get_known_companies()
    except Exception:
        logger.warning("Failed to fetch known companies; prompt will use empty list", exc_info=True)
        companies = []
    known_tickers = _format_known_tickers(companies)

    # Build the full system prompt
    full_prompt = _build_system_prompt(system_prompt, deps, known_tickers=known_tickers)

    # Load conversation history for multi-turn context
    message_history = await get_conversation_history(session_id) if session_id else None

    # Run the agent
    logger.info(
        f"Running agent query: '{query[:50]}...' "
        f"(session={session_id})"
    )

    settings = get_settings()
    usage_limits = UsageLimits(tool_calls_limit=settings.max_tool_calls)

    result = await kb_agent.run(
        query,
        deps=deps,
        message_history=message_history,
        instructions=full_prompt,
        usage_limits=usage_limits,
    )

    response = result.output
    response._raw_retrieved_chunks = deps.retrieved_chunks

    # Use actual retrieved chunks for sources (real chunk_ids) instead of LLM output
    # which may use indices like "1", "2" — frontend needs real UUIDs for get transcript
    response.sources = [
        SourceDocument(
            chunk_id=c["chunk_id"],
            content=c["content"],
            similarity=c.get("similarity", 0.0),
            metadata=c.get("metadata", {}),
        )
        for c in deps.retrieved_chunks
    ]
    # Resolve model citations to character spans for highlighting (no extra LLM call)
    if response.citations:
        _apply_citations_to_sources(response.sources, response.citations)

    # Persist new messages to conversation history
    if session_id:
        new_msgs = result.new_messages()
        if new_msgs:
            await append_conversation_messages(session_id, new_msgs)

    return response
