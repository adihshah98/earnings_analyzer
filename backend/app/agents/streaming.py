"""Simple RAG: streaming and non-streaming query flow."""

import asyncio
import logging
import re
import time
from collections import deque
from datetime import date
from typing import Any

from app.agents.prompt_utils import build_system_prompt, format_known_tickers
from app.agents.simple_rag import (
    _format_context_for_prompt,
    resolve_company_and_date,
)
from app.config import get_settings
from app.conversations.service import (
    append_conversation_turn,
    get_conversation_history,
    get_recent_user_queries,
)
from app.models.schemas import AgentResponse, SourceDocument
from app.prompts.templates import SIMPLE_RAG_SYSTEM_PROMPT
from app.rag.embeddings import generate_embedding, get_openai_client
from app.rag.retriever import (
    get_known_companies,
    retrieve_relevant_chunks,
)

logger = logging.getLogger(__name__)

# In-memory fallback: session_id → deque of recent user query strings (last 3).
# Written synchronously before fire-and-forget so the next query always sees prior context
# even if DB persistence hasn't completed yet.
_SESSION_RECENT_QUERIES: dict[str, deque[str]] = {}
_SESSION_RECENT_QUERIES_LIMIT = 4

_GREETING_RE = re.compile(
    r"^\s*(hey|hi+|hello|howdy|good\s+(morning|afternoon|evening)|what'?s\s+up|sup|yo|hiya|greetings|howzit)\s*[!?.,]?\s*$",
    re.IGNORECASE,
)
_GREETING_RESPONSE = (
    "Hi! I'm an earnings call analyst. Ask me anything about company earnings, revenue, "
    "guidance, or financial performance — for a specific company or across multiple companies. "
    "What would you like to know?"
)


def _is_greeting(query: str) -> bool:
    return bool(_GREETING_RE.match(query.strip()))


async def _append_conversation_turn_background(
    session_id: str, query: str, answer: str, user_id: str | None = None
) -> None:
    """Fire-and-forget: persist conversation turn without blocking the response."""
    try:
        await append_conversation_turn(session_id, query, answer, user_id=user_id)
    except Exception as e:
        logger.warning("Failed to persist conversation turn (background): %s", e)


def _build_search_query(query: str, conversation_context: list[str] | None) -> str:
    """Build search query with optional conversation context for retrieval."""
    if not conversation_context:
        return query
    context_str = " ".join(conversation_context[-3:])
    return f"Previous context: {context_str}. Current question: {query}".strip()


async def _prepare_simple_rag(
    query: str,
    search_mode: str | None,
    retrieval_threshold: float | None,
    session_id: str | None = None,
) -> tuple[list[dict[str, Any]], str, str]:
    """Shared setup: resolve scope, retrieve chunks, build prompt. Returns (chunks, full_prompt, today_iso)."""
    t_total = time.perf_counter()
    settings = get_settings()
    today_iso = date.today().isoformat()

    async def _get_companies():
        try:
            return await get_known_companies()
        except Exception:
            return []

    async def _get_message_history():
        if session_id:
            return await get_conversation_history(session_id)
        return []

    t0 = time.perf_counter()
    companies, message_history = await asyncio.gather(
        _get_companies(),
        _get_message_history(),
    )
    companies = companies or []
    logger.info("[latency] get_known_companies + get_conversation_history: %.3fs", time.perf_counter() - t0)

    db_queries = get_recent_user_queries(message_history, limit=4) if message_history else []
    # Merge with in-memory buffer to handle the fire-and-forget timing race:
    # the prior turn may not be in DB yet when this request arrives.
    mem_queries = list(_SESSION_RECENT_QUERIES.get(session_id, [])) if session_id else []
    merged = db_queries[:]
    for q in mem_queries:
        if q not in merged:
            merged.append(q)
    conversation_context = merged[-4:] if merged else None
    logger.info("[debug] conversation_context: %s", conversation_context)
    search_query = _build_search_query(query, conversation_context)

    # Run entity resolution and embedding in parallel to reduce latency
    t0 = time.perf_counter()
    scope, query_embedding = await asyncio.gather(
        resolve_company_and_date(query=query, companies=companies, today_iso=today_iso, conversation_context=conversation_context),
        generate_embedding(search_query),
    )
    logger.info("[latency] resolve_company_and_date + generate_embedding (parallel): %.3fs", time.perf_counter() - t0)

    filter_metadata = {"_resolved_date_pairs": scope.resolved_date_pairs} if scope.resolved_date_pairs else None

    top_k = settings.retrieval_top_k
    if scope.resolved_date_pairs:
        top_k = min(20, 5 * len(scope.resolved_date_pairs))

    effective_mode = search_mode or settings.default_search_mode
    threshold = (
        retrieval_threshold
        if retrieval_threshold is not None
        else settings.retrieval_threshold
    )

    t0 = time.perf_counter()
    chunks = await retrieve_relevant_chunks(
        query=query,
        top_k=top_k,
        threshold=threshold,
        filter_metadata=filter_metadata,
        conversation_context=conversation_context,
        search_mode=effective_mode,
        query_embedding=query_embedding,
    )
    logger.info("[latency] retrieve_relevant_chunks: %.3fs", time.perf_counter() - t0)

    context_str = _format_context_for_prompt(chunks)
    known_tickers = format_known_tickers(companies)
    full_prompt = build_system_prompt(
        SIMPLE_RAG_SYSTEM_PROMPT,
        context=context_str,
        known_tickers=known_tickers,
        today_date=today_iso,
    )
    logger.info("[latency] _prepare_simple_rag total: %.3fs", time.perf_counter() - t_total)
    return chunks, full_prompt, today_iso


_SOURCE_REF_RE = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)


def _parse_cited_source_indices(answer: str) -> set[int]:
    """Parse [Source N] markers from the answer; returns set of 1-based indices."""
    return {int(m.group(1)) for m in _SOURCE_REF_RE.finditer(answer)}


def _build_sources(
    chunks: list[dict[str, Any]], cited_indices: set[int]
) -> list[dict[str, Any]]:
    """Return source dicts filtered to cited chunks. Falls back to all chunks if none cited."""
    use_all = not cited_indices
    return [
        {
            "chunk_id": c["chunk_id"],
            "content": c.get("content", ""),
            "similarity": c.get("similarity", 0.0),
            "metadata": c.get("metadata", {}),
            "cited_spans": [],
        }
        for i, c in enumerate(chunks, 1)
        if use_all or i in cited_indices
    ]


def _confidence_from_chunks(chunks: list[dict[str, Any]], cited_indices: set[int]) -> float:
    """Mean similarity of cited chunks; falls back to all chunks if none cited."""
    relevant = [c for i, c in enumerate(chunks, 1) if not cited_indices or i in cited_indices]
    if not relevant:
        return 0.0
    return round(sum(c.get("similarity", 0.0) for c in relevant) / len(relevant), 3)


async def run_simple_rag(
    query: str,
    search_mode: str | None = None,
    retrieval_threshold: float | None = None,
    session_id: str | None = None,
) -> AgentResponse:
    """Run the simple RAG flow (non-streaming). Same as prod. Returns AgentResponse."""
    chunks, full_prompt, _ = await _prepare_simple_rag(
        query=query,
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

    cited_indices = _parse_cited_source_indices(answer)
    sources = [
        SourceDocument(
            chunk_id=c["chunk_id"],
            content=c.get("content", ""),
            similarity=c.get("similarity", 0.0),
            metadata=c.get("metadata", {}),
        )
        for i, c in enumerate(chunks, 1)
        if not cited_indices or i in cited_indices
    ]
    return AgentResponse(
        answer=answer,
        confidence=_confidence_from_chunks(chunks, cited_indices),
        sources=sources,
        citations=[],
        reasoning=None,
        tool_calls_made=[],
    )


async def stream_simple_rag_or_agent(
    query: str,
    session_id: str | None,
    search_mode: str | None,
    retrieval_threshold: float | None,
    user_id: str | None = None,
):
    """Async generator: yields ('delta', text) for token chunks, then ('done', payload dict)."""
    if _is_greeting(query):
        if session_id:
            buf = _SESSION_RECENT_QUERIES.setdefault(session_id, deque(maxlen=_SESSION_RECENT_QUERIES_LIMIT))
            buf.append(query)
            asyncio.create_task(
                _append_conversation_turn_background(session_id, query, _GREETING_RESPONSE, user_id=user_id)
            )
        yield "delta", _GREETING_RESPONSE
        yield "done", {
            "answer": _GREETING_RESPONSE,
            "confidence": 1.0,
            "sources": [],
            "citations": [],
            "reasoning": None,
            "tool_calls_made": [],
        }
        return

    t_prepare = time.perf_counter()
    chunks, full_prompt, _ = await _prepare_simple_rag(
        query=query,
        search_mode=search_mode,
        retrieval_threshold=retrieval_threshold,
        session_id=session_id,
    )
    logger.info("[latency] prepare (to first LLM call): %.3fs", time.perf_counter() - t_prepare)

    client = get_openai_client()
    settings = get_settings()
    model = settings.openai_model
    # Use Responses API (recommended); input is list of messages, content is array of input_text items
    input_messages = [
        {"role": "system", "content": [{"type": "input_text", "text": full_prompt}]},
        {"role": "user", "content": [{"type": "input_text", "text": query}]},
    ]
    accumulated = []
    first_token = True
    t_ttft: float | None = None
    try:
        t_llm = time.perf_counter()
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
                    if first_token:
                        t_ttft = time.perf_counter() - t_llm
                        logger.info("[latency] LLM time_to_first_token: %.3fs", t_ttft)
                        first_token = False
                    accumulated.append(delta)
                    yield "delta", delta
    except Exception as e:
        logger.exception("Streaming completion failed: %s", e)
        accumulated = ["[Error while streaming response.]"]
    else:
        logger.info(
            "[latency] LLM stream total: %.3fs (tokens: %d)",
            time.perf_counter() - t_llm,
            len(accumulated),
        )

    answer = "".join(accumulated)
    cited_indices = _parse_cited_source_indices(answer)
    sources = _build_sources(chunks, cited_indices)
    if session_id:
        # Update in-memory buffer synchronously so the next query sees this turn immediately,
        # regardless of when the async DB write completes.
        buf = _SESSION_RECENT_QUERIES.setdefault(session_id, deque(maxlen=_SESSION_RECENT_QUERIES_LIMIT))
        buf.append(query)
        asyncio.create_task(
            _append_conversation_turn_background(session_id, query, answer, user_id=user_id)
        )
    yield "done", {
        "answer": answer,
        "confidence": _confidence_from_chunks(chunks, cited_indices),
        "sources": sources,
        "citations": [],
        "reasoning": None,
        "tool_calls_made": [],
    }
