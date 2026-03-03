"""Simple RAG: streaming and non-streaming query flow."""

import asyncio
import logging
from datetime import date
from typing import Any

from app.agents.prompt_utils import build_system_prompt, format_known_tickers
from app.agents.simple_rag import (
    _format_context_for_prompt,
    resolve_company_and_date,
)
from app.config import get_settings
from app.conversations.service import get_conversation_history, get_recent_user_queries
from app.models.schemas import AgentResponse, SourceDocument
from app.prompts.templates import SIMPLE_RAG_SYSTEM_PROMPT
from app.rag.embeddings import get_openai_client
from app.rag.retriever import get_known_companies, retrieve_relevant_chunks

logger = logging.getLogger(__name__)


async def _prepare_simple_rag(
    query: str,
    request_company_ticker: str | None,
    request_as_of_date: str | None,
    search_mode: str | None,
    retrieval_threshold: float | None,
    session_id: str | None = None,
) -> tuple[list[dict[str, Any]], str, str]:
    """Shared setup: resolve scope, retrieve chunks, build prompt. Returns (chunks, full_prompt, today_iso)."""
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

    filter_metadata: dict[str, Any] = {}
    if scope.resolved_date_pairs:
        filter_metadata["_resolved_date_pairs"] = scope.resolved_date_pairs
    else:
        if scope.company_ticker:
            filter_metadata["company_ticker"] = scope.company_ticker
        if scope.as_of_date:
            filter_metadata["as_of_date"] = scope.as_of_date
    filter_metadata = filter_metadata or None

    top_k = settings.retrieval_top_k
    if scope.resolved_date_pairs:
        top_k = min(20, 5 * len(scope.resolved_date_pairs))

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
        top_k=top_k,
        threshold=threshold,
        filter_metadata=filter_metadata,
        conversation_context=conversation_context,
        search_mode=effective_mode,
    )

    context_str = _format_context_for_prompt(chunks)
    known_tickers = format_known_tickers(companies)
    full_prompt = build_system_prompt(
        SIMPLE_RAG_SYSTEM_PROMPT,
        context=context_str,
        known_tickers=known_tickers,
        today_date=today_iso,
    )
    return chunks, full_prompt, today_iso


async def run_simple_rag(
    query: str,
    request_company_ticker: str | None = None,
    request_as_of_date: str | None = None,
    search_mode: str | None = None,
    retrieval_threshold: float | None = None,
    session_id: str | None = None,
) -> AgentResponse:
    """Run the simple RAG flow (non-streaming). Same as prod. Returns AgentResponse."""
    chunks, full_prompt, _ = await _prepare_simple_rag(
        query=query,
        request_company_ticker=request_company_ticker,
        request_as_of_date=request_as_of_date,
        search_mode=search_mode,
        retrieval_threshold=retrieval_threshold,
        session_id=session_id,
    )

    client = get_openai_client()
    settings = get_settings()
    input_messages = [
        {"role": "system", "content": [{"type": "input_text", "text": full_prompt}]},
        {"role": "user", "content": [{"type": "input_text", "text": query}]},
    ]

    try:
        response = await client.responses.create(
            model=settings.openai_model,
            input=input_messages,
        )
        raw = (
            response.output_text
            if hasattr(response, "output_text")
            else ""
        )
        answer = (raw or "").strip() or "[No response generated.]"
    except Exception as e:
        logger.exception("Simple RAG completion failed: %s", e)
        answer = "[Error while generating response.]"

    sources = [
        SourceDocument(
            chunk_id=c["chunk_id"],
            content=c["content"],
            similarity=c.get("similarity", 0.0),
            metadata=c.get("metadata", {}),
        )
        for c in chunks
    ]
    return AgentResponse(
        answer=answer,
        confidence=0.9,
        sources=sources,
        citations=[],
        reasoning=None,
        tool_calls_made=[],
    )


async def stream_simple_rag_or_agent(
    query: str,
    session_id: str | None,
    request_company_ticker: str | None,
    request_as_of_date: str | None,
    search_mode: str | None,
    retrieval_threshold: float | None,
):
    """Async generator: yields ('delta', text) for token chunks, then ('done', payload dict)."""
    chunks, full_prompt, _ = await _prepare_simple_rag(
        query=query,
        request_company_ticker=request_company_ticker,
        request_as_of_date=request_as_of_date,
        search_mode=search_mode,
        retrieval_threshold=retrieval_threshold,
        session_id=session_id,
    )

    client = get_openai_client()
    settings = get_settings()
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
