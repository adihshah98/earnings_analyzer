"""Streaming response for agent query: SSE with token deltas and final payload."""

import asyncio
import json
import logging
from typing import Any

from app.agents.simple_rag import (
    SimpleRAGScope,
    _format_context_for_prompt,
    resolve_company_and_date,
)
from app.config import get_settings
from app.conversations.service import get_conversation_history, get_recent_user_queries
from app.rag.retriever import get_known_companies, retrieve_relevant_chunks
from app.agents.kb_agent import (
    _build_system_prompt,
    _format_known_tickers,
    query_agent,
)
from app.prompts.templates import SIMPLE_RAG_SYSTEM_PROMPT
from app.tools.agent_tools import AgentDeps
from datetime import date

logger = logging.getLogger(__name__)


async def stream_simple_rag_or_agent(
    query: str,
    session_id: str | None,
    request_company_ticker: str | None,
    request_as_of_date: str | None,
    search_mode: str | None,
    retrieval_threshold: float | None,
):
    """Async generator: yields ('delta', text) for token chunks, then ('done', payload dict)."""
    settings = get_settings()
    today_iso = date.today().isoformat()

    async def _get_companies():
        try:
            return await get_known_companies()
        except Exception:
            return []

    companies, message_history = await asyncio.gather(
        _get_companies(),
        get_conversation_history(session_id) if session_id else [],
    )
    companies = companies or []

    scope = await resolve_company_and_date(
        query=query,
        request_company_ticker=request_company_ticker,
        request_as_of_date=request_as_of_date,
        companies=companies,
        today_iso=today_iso,
    )
    if not scope.use_simple_path:
        response = await query_agent(
            query=query,
            session_id=session_id,
            search_mode=search_mode,
            retrieval_threshold=retrieval_threshold,
            filter_metadata=(
                {"company_ticker": request_company_ticker}
                if request_company_ticker
                else None
            ),
        )
        yield "done", response.model_dump(mode="json")
        return

    filter_metadata: dict[str, Any] = {}
    if scope.company_ticker:
        filter_metadata["company_ticker"] = scope.company_ticker
    if scope.as_of_date:
        filter_metadata["as_of_date"] = scope.as_of_date
    filter_metadata = filter_metadata or None

    conversation_context = (
        get_recent_user_queries(message_history, limit=3) if message_history else None
    )
    effective_mode = search_mode or settings.default_search_mode
    threshold = (
        retrieval_threshold
        if retrieval_threshold is not None
        else settings.retrieval_threshold
    )

    chunks = await retrieve_relevant_chunks(
        query=query,
        threshold=threshold,
        filter_metadata=filter_metadata,
        conversation_context=conversation_context,
        search_mode=effective_mode,
    )
    if not chunks:
        response = await query_agent(
            query=query,
            session_id=session_id,
            search_mode=search_mode,
            retrieval_threshold=retrieval_threshold,
            filter_metadata=filter_metadata,
        )
        yield "done", response.model_dump(mode="json")
        return

    context_str = _format_context_for_prompt(chunks)
    known_tickers = _format_known_tickers(companies)
    deps = AgentDeps(
        session_id=session_id,
        filter_metadata=filter_metadata or {},
        today_date=today_iso,
        search_mode=effective_mode,
        retrieval_threshold=threshold,
    )
    full_prompt = _build_system_prompt(
        SIMPLE_RAG_SYSTEM_PROMPT,
        deps,
        context=context_str,
        known_tickers=known_tickers,
    )

    from app.rag.embeddings import get_openai_client

    client = get_openai_client()
    model = settings.openai_model
    # Use Responses API (recommended); input is list of messages, content is array of input_text items
    input_messages = [
        {"role": "system", "content": [{"type": "input_text", "text": full_prompt}]},
        {"role": "user", "content": [{"type": "input_text", "text": query}]},
    ]
    accumulated = []
    try:
        stream = await client.responses.create(
            model=model,
            input=input_messages,
            stream=True,
        )
        async for event in stream:
            event_type = getattr(event, "type", None) or (
                event.get("type") if isinstance(event, dict) else None
            )
            if event_type in ("response.output_text.delta", "response.output_text_delta"):
                delta = getattr(event, "delta", None) or (
                    event.get("delta") if isinstance(event, dict) else None
                )
                if delta:
                    accumulated.append(delta)
                    yield "delta", delta
    except Exception as e:
        logger.exception("Streaming completion failed: %s", e)
        accumulated = ["[Error while streaming response.]"]

    sources = [
        {
            "chunk_id": c["chunk_id"],
            "content": c["content"],
            "similarity": c.get("similarity", 0.0),
            "metadata": c.get("metadata", {}),
            "cited_spans": [],
        }
        for c in chunks
    ]
    answer = "".join(accumulated)
    payload = {
        "answer": answer,
        "confidence": 0.9,
        "sources": sources,
        "citations": [],
        "reasoning": None,
        "tool_calls_made": [],
    }
    yield "done", payload
