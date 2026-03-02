"""Main PydanticAI knowledge base agent."""

import logging
from typing import Any

from pydantic_ai import Agent, UsageLimits

from app.config import get_settings
from app.conversations.service import append_conversation_messages, get_conversation_history
from app.models.schemas import AgentResponse
from app.prompts.templates import SYSTEM_PROMPT_V1
from app.rag.retriever import get_known_companies
from app.tools.agent_tools import (
    AgentDeps,
    calculate,
    list_available_transcripts,
    search_knowledge_base,
)

logger = logging.getLogger(__name__)


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

    try:
        return template.format(
            context=context,
            user_name=user_name,
            user_role=user_role,
            user_department=user_department,
            custom_instructions=custom_instructions,
            known_tickers=known_tickers,
        )
    except KeyError:
        return template.replace("{context}", context).replace(
            "{known_tickers}", known_tickers
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

    # Persist new messages to conversation history
    if session_id:
        new_msgs = result.new_messages()
        if new_msgs:
            await append_conversation_messages(session_id, new_msgs)

    return response
