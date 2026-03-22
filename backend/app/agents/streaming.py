"""Simple RAG: streaming and non-streaming query flow."""

import asyncio
import logging
import re
import time
from collections import deque
from datetime import date
from typing import Any

from cachetools import TTLCache

from app.agents.prompt_utils import build_system_prompt, format_known_tickers
from app.agents.simple_rag import (
    _format_context_for_prompt,
    _rewrite_query_for_retrieval,
    build_resolution_note,
    reorder_chunks_for_range,
    resolve_company_and_date,
    rewrite_query_for_answer_llm,
    trim_chunks_to_token_budget,
)
from app.config import get_settings
from app.conversations.service import (
    append_conversation_turn,
    get_conversation_history,
    get_recent_turns,
)
from app.models.schemas import AgentResponse, SourceDocument
from app.prompts.templates import SIMPLE_RAG_SYSTEM_PROMPT
from app.rag.embeddings import generate_embedding, get_openai_client
from app.rag.retriever import (
    get_available_periods,
    get_financials_chunks_for_pairs,
    get_known_companies,
    retrieve_relevant_chunks,
)

logger = logging.getLogger(__name__)

# In-memory fallback: session_id → deque of recent "Q: ... A: ..." turn strings (last 5).
# Written synchronously before fire-and-forget so the next query always sees prior context
# even if DB persistence hasn't completed yet.
# TTLCache evicts idle sessions after 2h to prevent unbounded memory growth.
_SESSION_RECENT_QUERIES: TTLCache = TTLCache(maxsize=2000, ttl=7200)
_SESSION_RECENT_QUERIES_LIMIT = 5

# Strong references to fire-and-forget tasks so they aren't garbage-collected.
_background_tasks: set[asyncio.Task] = set()


def _fire_and_forget(coro) -> None:
    """Schedule a coroutine as a background task with prevented GC."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


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


def _extract_prior_turns(message_history: list, limit: int = 3) -> list[dict]:
    """Convert ModelMessage history to the last `limit` (user, assistant) turn pairs
    in OpenAI Responses API input format."""
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

    pairs: list[tuple[str, str]] = []
    msgs = list(message_history)
    i = 0
    while i < len(msgs):
        msg = msgs[i]
        if isinstance(msg, ModelRequest):
            user_text = " ".join(
                p.content for p in msg.parts
                if isinstance(p, UserPromptPart) and isinstance(p.content, str)
            ).strip()
            assistant_text = ""
            if i + 1 < len(msgs) and isinstance(msgs[i + 1], ModelResponse):
                assistant_text = " ".join(
                    p.content for p in msgs[i + 1].parts if isinstance(p, TextPart)
                ).strip()
                i += 2
            else:
                i += 1
            if user_text:
                pairs.append((user_text, assistant_text))
        else:
            i += 1

    result = []
    for user_text, assistant_text in pairs[-limit:]:
        result.append({"role": "user", "content": [{"type": "input_text", "text": user_text}]})
        if assistant_text:
            result.append({"role": "assistant", "content": [{"type": "output_text", "text": assistant_text}]})
    return result


async def _prepare_simple_rag(
    query: str,
    search_mode: str | None,
    retrieval_threshold: float | None,
    session_id: str | None = None,
) -> tuple[list[dict[str, Any]], str, str, list[dict], str]:
    """Shared setup: resolve scope, retrieve chunks, build prompt. Returns (chunks, full_prompt, today_iso, prior_turns, llm_query)."""
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

    async def _get_periods():
        try:
            return await get_available_periods()
        except Exception:
            return {}

    t0 = time.perf_counter()
    companies, message_history, available_periods = await asyncio.gather(
        _get_companies(),
        _get_message_history(),
        _get_periods(),
    )
    companies = companies or []
    logger.info("[latency] get_known_companies + history + periods: %.3fs", time.perf_counter() - t0)

    db_turns = get_recent_turns(message_history, limit=5) if message_history else []
    db_context = [f"Q: {q} A: {a}" for q, a in db_turns]
    # Merge with in-memory buffer to handle the fire-and-forget timing race:
    # the prior turn may not be in DB yet when this request arrives.
    mem_context = list(_SESSION_RECENT_QUERIES.get(session_id, [])) if session_id else []
    merged = db_context[:]
    for item in mem_context:
        if item not in merged:
            merged.append(item)
    conversation_context = merged[-5:] if merged else None
    logger.info("[debug] conversation_context: %s", conversation_context)

    # Run scope resolution (tickers + temporal), query rewriting, and embedding in parallel.
    t0 = time.perf_counter()
    scope, rewritten_queries, query_embedding = await asyncio.gather(
        resolve_company_and_date(
            query=query, companies=companies, today_iso=today_iso,
            conversation_context=conversation_context,
            available_periods=available_periods,
        ),
        _rewrite_query_for_retrieval(query),
        generate_embedding(query),
    )
    logger.info("[latency] resolve_scope + rewrite + embedding (parallel): %.3fs", time.perf_counter() - t0)

    # Ensure the original query is always in the retrieval set so its
    # pre-computed embedding is actually used (rewrites rarely match it verbatim).
    if query not in rewritten_queries:
        rewritten_queries.insert(0, query)
    logger.info("[debug] rewritten_queries: %s", rewritten_queries)

    # Build filter_metadata: ticker_date_pairs is most specific, then tickers-only
    if scope.ticker_date_pairs:
        filter_metadata = {"_ticker_date_pairs": scope.ticker_date_pairs}
    elif scope.tickers:
        filter_metadata = {"_tickers": scope.tickers}
    else:
        filter_metadata = None

    effective_mode = search_mode or settings.default_search_mode
    threshold = retrieval_threshold if retrieval_threshold is not None else settings.retrieval_threshold
    top_k = 40 if scope.tickers else settings.retrieval_top_k

    # Retrieve for each rewritten query in parallel; original query reuses its pre-computed embedding.
    t0 = time.perf_counter()
    per_query_chunks = await asyncio.gather(*[
        retrieve_relevant_chunks(
            query=q,
            top_k=top_k,
            threshold=threshold,
            filter_metadata=filter_metadata,
            search_mode=effective_mode,
            query_embedding=query_embedding if q == query else None,
        )
        for q in rewritten_queries
    ])
    logger.info("[latency] retrieve_relevant_chunks (%d queries): %.3fs", len(rewritten_queries), time.perf_counter() - t0)

    # Merge: deduplicate by chunk_id, keep highest similarity score, re-sort.
    seen: dict[str, dict] = {}
    for result_chunks in per_query_chunks:
        for chunk in result_chunks:
            cid = chunk["chunk_id"]
            if cid not in seen or chunk["similarity"] > seen[cid]["similarity"]:
                seen[cid] = chunk
    chunks = sorted(seen.values(), key=lambda x: x["similarity"], reverse=True)[:top_k]

    # Always include financial summary chunks for resolved (ticker, call_date) pairs.
    # When temporal resolution succeeded, use all resolved pairs; otherwise extract from results.
    if scope.ticker_date_pairs:
        covered_pairs = list(scope.ticker_date_pairs)
    else:
        covered_pairs = []
        seen_pairs: set[tuple[str, str]] = set()
        for chunk in chunks:
            meta = chunk.get("metadata") or {}
            t = meta.get("company_ticker", "")
            d = meta.get("call_date", "")
            if t and d and (t, d) not in seen_pairs:
                seen_pairs.add((t, d))
                covered_pairs.append((t, d))

    if covered_pairs:
        t0 = time.perf_counter()
        financials_chunks = await get_financials_chunks_for_pairs(covered_pairs)
        logger.info("[latency] get_financials_chunks_for_pairs (%d pairs): %.3fs", len(covered_pairs), time.perf_counter() - t0)
        existing_ids = {c["chunk_id"] for c in chunks}
        for fc in financials_chunks:
            if fc["chunk_id"] not in existing_ids:
                chunks.append(fc)

    chunks = trim_chunks_to_token_budget(chunks)
    chunks = reorder_chunks_for_range(chunks, scope)
    context_str = _format_context_for_prompt(chunks)
    resolution_note = build_resolution_note(scope, available_periods)
    if resolution_note:
        context_str = resolution_note + "\n\n" + context_str
    known_tickers = format_known_tickers(companies)
    full_prompt = build_system_prompt(
        SIMPLE_RAG_SYSTEM_PROMPT,
        context=context_str,
        known_tickers=known_tickers,
        today_date=today_iso,
    )
    llm_query = rewrite_query_for_answer_llm(query, scope, available_periods)
    if llm_query != query:
        logger.info("[debug] query rewritten for answer LLM: %r → %r", query, llm_query)
    prior_turns = _extract_prior_turns(message_history, limit=3)
    logger.info("[latency] _prepare_simple_rag total: %.3fs", time.perf_counter() - t_total)
    return chunks, full_prompt, today_iso, prior_turns, llm_query


_SOURCE_REF_RE = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)


def _parse_cited_source_indices(answer: str) -> set[int]:
    """Parse [Source N] markers from the answer; returns set of 1-based indices."""
    return {int(m.group(1)) for m in _SOURCE_REF_RE.finditer(answer)}


def _normalize_source_refs(
    answer: str, sources: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    """Renumber [Source N] citations in the answer to 1..n in order of first appearance.

    Updates source_index on each source dict to match the new numbers so the
    source cards and the answer text stay in sync.
    """
    norm: dict[int, int] = {}

    def _replace(m: re.Match) -> str:
        orig = int(m.group(1))
        if orig not in norm:
            norm[orig] = len(norm) + 1
        return f"[Source {norm[orig]}]"

    normalized_answer = _SOURCE_REF_RE.sub(_replace, answer)
    for s in sources:
        if s.get("source_index") in norm:
            s["source_index"] = norm[s["source_index"]]
    return normalized_answer, sources


def _build_sources(
    chunks: list[dict[str, Any]], cited_indices: set[int]
) -> list[dict[str, Any]]:
    """Return only chunks explicitly cited in the answer via [Source N] markers.

    source_index preserves the original 1-based position so the UI label matches
    the [Source N] citation in the answer text.
    """
    return [
        {
            "chunk_id": c["chunk_id"],
            "content": c.get("content", ""),
            "similarity": c.get("similarity", 0.0),
            "metadata": c.get("metadata", {}),
            "cited_spans": [],
            "source_index": i,
        }
        for i, c in enumerate(chunks, 1)
        if i in cited_indices
    ]



async def run_simple_rag(
    query: str,
    search_mode: str | None = None,
    retrieval_threshold: float | None = None,
    session_id: str | None = None,
) -> AgentResponse:
    """Run the simple RAG flow (non-streaming). Same as prod. Returns AgentResponse."""
    chunks, full_prompt, _, prior_turns, llm_query = await _prepare_simple_rag(
        query=query,
        search_mode=search_mode,
        retrieval_threshold=retrieval_threshold,
        session_id=session_id,
    )

    client = get_openai_client()
    settings = get_settings()
    input_messages = [
        {"role": "system", "content": [{"type": "input_text", "text": full_prompt}]},
        *prior_turns,
        {"role": "user", "content": [{"type": "input_text", "text": llm_query}]},
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
    raw_sources = _build_sources(chunks, cited_indices)
    answer, raw_sources = _normalize_source_refs(answer, raw_sources)
    sources = [SourceDocument(**s) for s in raw_sources]
    return AgentResponse(
        answer=answer,
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
            buf.append(f"Q: {query} A: {_GREETING_RESPONSE}")
            _fire_and_forget(
                _append_conversation_turn_background(session_id, query, _GREETING_RESPONSE, user_id=user_id)
            )
        yield "delta", _GREETING_RESPONSE
        yield "done", {
            "answer": _GREETING_RESPONSE,
            "sources": [],
            "citations": [],
            "reasoning": None,
            "tool_calls_made": [],
        }
        return

    t_prepare = time.perf_counter()
    chunks, full_prompt, _, prior_turns, llm_query = await _prepare_simple_rag(
        query=query,
        search_mode=search_mode,
        retrieval_threshold=retrieval_threshold,
        session_id=session_id,
    )
    logger.info("[latency] prepare (to first LLM call): %.3fs", time.perf_counter() - t_prepare)

    client = get_openai_client()
    settings = get_settings()
    model = settings.openai_model
    # Use Responses API (recommended); prior_turns injects conversation history so the LLM
    # can resolve pronouns and provide attribution across multi-turn sessions.
    input_messages = [
        {"role": "system", "content": [{"type": "input_text", "text": full_prompt}]},
        *prior_turns,
        {"role": "user", "content": [{"type": "input_text", "text": llm_query}]},
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
    answer, sources = _normalize_source_refs(answer, sources)
    if session_id:
        # Update in-memory buffer synchronously so the next query sees this turn immediately,
        # regardless of when the async DB write completes.
        buf = _SESSION_RECENT_QUERIES.setdefault(session_id, deque(maxlen=_SESSION_RECENT_QUERIES_LIMIT))
        buf.append(f"Q: {query} A: {answer[:400]}")
        _fire_and_forget(
            _append_conversation_turn_background(session_id, query, answer, user_id=user_id)
        )
    yield "done", {
        "answer": answer,
        "sources": sources,
        "citations": [],
        "reasoning": None,
        "tool_calls_made": [],
    }
