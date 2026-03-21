"""Simple RAG: single retrieval + one LLM call (no agentic tool loop).

Entity (company) and date are resolved via a cheap LLM call returning a list of
(entity, date) pairs. All queries use this path.
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from cachetools import TTLCache

from app.config import get_settings
from app.conversations.service import get_recent_user_queries
from app.rag.embeddings import get_openai_client
from app.rag.retriever import (
    resolve_entities_to_call_dates,
    resolve_entities_to_call_dates_in_ranges,
    retrieve_relevant_chunks,
)

logger = logging.getLogger(__name__)

# Entity resolution scope cache (24h TTL). Key: (query_normalized, today_iso, companies_key).
_SCOPE_CACHE: TTLCache = TTLCache(maxsize=500, ttl=86400)

# Quarter end dates (approximate): Q1 -> Mar 31, Q2 -> Jun 30, Q3 -> Sep 30, Q4 -> Dec 31
_QUARTER_END = {"1": "03-31", "2": "06-30", "3": "09-30", "4": "12-31"}

_RESOLUTION_MAX_TOKENS = 300  # increased to handle date_range objects

# (company_ticker, date_expression, date_range_start, date_range_end)
# date_range_start/end are ISO strings for range queries; date_expression for single-period queries.
ResolvedEntity = tuple[str, str | None, str | None, str | None]


def _extract_text_from_responses_output(output: list) -> str:
    """Extract text from Responses API output (fallback when output_text unavailable)."""
    parts = []
    for item in output or []:
        if getattr(item, "type", None) == "message" and hasattr(item, "content"):
            for c in item.content or []:
                if getattr(c, "type", None) == "output_text":
                    parts.append(getattr(c, "text", "") or "")
    return "".join(parts)


@dataclass
class SimpleRAGScope:
    """Resolved scope for retrieval.

    Always expressed as pre-resolved (ticker, call_date) pairs so the retriever uses
    a single OR-filter regardless of how many companies or time periods are involved.
    None means no filter (full-corpus search).
    """

    resolved_date_pairs: list[tuple[str, str]] | None = None


def _date_expression_to_iso(date_expression: str | None, today_iso: str) -> str | None:
    """Map LLM date_expression to ISO date. Returns None if unparseable."""
    if not date_expression or not date_expression.strip():
        return None
    expr = date_expression.strip().lower()
    # Already YYYY-MM-DD
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", expr):
        return expr
    # last_quarter, current_quarter
    try:
        y, m, d = map(int, today_iso.split("-"))
        if expr in ("last_quarter", "previous_quarter", "prior_quarter"):
            # Previous quarter end: if we're in Q1 (Jan–Mar) -> last year Q4; else same year
            if m <= 3:
                return f"{y - 1}-12-31"
            if m <= 6:
                return f"{y}-03-31"
            if m <= 9:
                return f"{y}-06-30"
            return f"{y}-09-30"
        if expr in ("current_quarter", "this_quarter"):
            if m <= 3:
                return f"{y}-03-31"
            if m <= 6:
                return f"{y}-06-30"
            if m <= 9:
                return f"{y}-09-30"
            return f"{y}-12-31"
    except (ValueError, AttributeError):
        pass
    # Q1 2024, Q2 2023
    q_match = re.search(r"q([1-4])\s*['\"]?\s*(\d{4})", expr, re.IGNORECASE)
    if q_match:
        q, year = q_match.group(1), q_match.group(2)
        return f"{year}-{_QUARTER_END.get(q, '12-31')}"
    return None


async def _resolve_entities_via_llm(
    query: str,
    companies: list[dict[str, str]],
    today_iso: str,
) -> list[ResolvedEntity]:
    """Call a cheap fast LLM to extract a list of (company_ticker, date_expression) pairs.

    Returns a list of (ticker, date_expression) for each entity the user is asking about.
    Single-entity queries return one element; multi-entity/comparison return multiple.
    Only tickers from the given companies list are included. date_expression is mapped to ISO later.
    """
    if not companies:
        return []
    valid_tickers = {c.get("ticker", "").upper() for c in companies if c.get("ticker")}
    company_list = ", ".join(f"{c.get('ticker', '').upper()}: {c.get('name', '')}" for c in companies if c.get("ticker"))
    # Precompute range helpers for the prompt
    try:
        y, m, _ = map(int, today_iso.split("-"))
        last_year_start = f"{y - 1}-01-01"
        last_year_end = f"{y - 1}-12-31"
        # Trailing 12 months start
        trailing_start_y, trailing_start_m = (y - 1, m) if m > 1 else (y - 2, 12)
        trailing_12_start = f"{trailing_start_y}-{trailing_start_m:02d}-01"
        # Last N quarters: start = first day of quarter N quarters before current quarter
        def _quarter_start_n_ago(n: int) -> str:
            current_q = (m - 1) // 3  # 0=Q1,1=Q2,2=Q3,3=Q4
            total_q = y * 4 + current_q - n
            qy, qq = total_q // 4, total_q % 4
            return f"{qy}-{qq * 3 + 1:02d}-01"
        last_4q_start = _quarter_start_n_ago(4)
        last_2q_start = _quarter_start_n_ago(2)
    except (ValueError, AttributeError):
        last_year_start, last_year_end, trailing_12_start = f"{today_iso[:4]}-01-01", f"{today_iso[:4]}-12-31", today_iso
        last_4q_start = last_2q_start = f"{today_iso[:4]}-01-01"

    prompt = f"""Today's date is {today_iso}.

Available companies (ticker: name): {company_list}

From the user query, extract a list of company/date entities the user is asking about.
Each object has: "company_ticker", and EITHER "date_expression" (single period) OR "date_range" (multi-period).

SINGLE-PERIOD rules (use "date_expression", set "date_range": null):
- Single company / single period: [{{"company_ticker": "ACME", "date_expression": "last_quarter", "date_range": null}}]
- Comparing multiple companies for same period: [{{"company_ticker": "ACME", "date_expression": "last_quarter", "date_range": null}}, {{"company_ticker": "GLOBEX", "date_expression": "last_quarter", "date_range": null}}]
- Same company, two specific periods (QoQ, YoY): [{{"company_ticker": "NOW", "date_expression": "last_quarter", "date_range": null}}, {{"company_ticker": "NOW", "date_expression": "current_quarter", "date_range": null}}]
- date_expression values: YYYY-MM-DD, "Q1 2024" style, "last_quarter", "current_quarter", or null for latest (most recent call only).
- DEFAULT (no date mentioned, no prior date context): use date_expression: null — this returns only the MOST RECENT call. NEVER default to a date_range just because no date is specified.

MULTI-PERIOD rules (use "date_range", set "date_expression": null) — use when the query spans multiple quarters or years.
Trigger this for ANY expression implying more than one period: exact counts ("last 4 quarters"), vague counts ("few quarters", "couple of quarters", "several years", "a few years"), named periods ("last year", "FY2024", "H1 2025"), or open-ended ("over time", "historically", "trailing 12 months").
Vague quantity defaults: "couple" = 2, "few" = 3, "several" = 5.
Quarter boundaries: Q1 = Jan 1, Q2 = Apr 1, Q3 = Jul 1, Q4 = Oct 1. Today is {today_iso}.

Examples:
- "last year" → {{"start": "{last_year_start}", "end": "{last_year_end}"}}
- "trailing 12 months" / "past 12 months" → {{"start": "{trailing_12_start}", "end": "{today_iso}"}}
- "all of 2024" / "FY2024" → {{"start": "2024-01-01", "end": "2024-12-31"}}
- "H1 2025" → {{"start": "2025-01-01", "end": "2025-06-30"}}
- "last 4 quarters" → {{"start": "{last_4q_start}", "end": "{today_iso}"}}
- "last 2 quarters" / "couple of quarters" → {{"start": "{last_2q_start}", "end": "{today_iso}"}}
- "last N quarters" → same pattern: go back N quarters from today's quarter, take the first day of that quarter as start
- "few years" (≈3) → {{"start": "{y - 3}-01-01", "end": "{today_iso}"}}
- For comparisons over a range, give ALL companies the SAME date_range.
  e.g. "Compare ACME and GLOBEX over the last year": [{{"company_ticker": "ACME", "date_expression": null, "date_range": {{"start": "{last_year_start}", "end": "{last_year_end}"}}}}, {{"company_ticker": "GLOBEX", "date_expression": null, "date_range": {{"start": "{last_year_start}", "end": "{last_year_end}"}}}}]

Use only tickers from the list above. Company matching must be fuzzy — match on approximate name, common abbreviations, misspellings, and partial names (e.g. "SrviceNOw", "servicenow", "SRVICENOW", "Service Now" all match NOW: ServiceNow).

IMPORTANT — resolve company AND date from conversation context:
1. Company: If the query uses pronouns ("their", "them", "it", "they", "the company") without naming a company, resolve using the conversation context.
2. Date: Only inherit a prior date when the current query is a clear follow-up (uses pronouns, no company named, or phrases like "what about X"). If the query explicitly names a company but mentions no date, use date_expression: null (most recent call) — do NOT inherit a prior date_range.
   Example: prior query had date_range {{"start": "{last_4q_start}", "end": "{today_iso}"}}, current query "Do they mention ACV?" → follow-up, inherit that date_range.
   Example: prior query "ServiceNow revenue last 4 quarters", current query "What about their NRR" → follow-up with pronoun, inherit date_range → [{{"company_ticker": "NOW", "date_expression": null, "date_range": {{"start": "{last_4q_start}", "end": "{today_iso}"}}}}]
   Example: prior query "ServiceNow revenue last 4 quarters", current query "ServiceNow revenue" → explicitly names company, no date → use date_expression: null (most recent only), do NOT inherit the range.

Only return [] if there is truly no company identifiable from either the query or the conversation context.

Reply with only a JSON object: {{ "entities": [ {{ "company_ticker": "<ticker>", "date_expression": "<or null>", "date_range": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}} or null }}, ... ] }}

User query: {query}"""
    try:
        client = get_openai_client()
        input_messages = [
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": "You output only valid JSON with key 'entities' (array of objects with company_ticker and date_expression)."},
                ],
            },
            {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
        ]
        t0 = time.perf_counter()
        response = await client.responses.create(
            model=get_settings().openai_model,
            input=input_messages,
            max_output_tokens=_RESOLUTION_MAX_TOKENS,
        )
        logger.info("[latency] _resolve_entities_via_llm: %.3fs", time.perf_counter() - t0)
        raw = (
            response.output_text
            if hasattr(response, "output_text")
            else _extract_text_from_responses_output(getattr(response, "output", []))
        )
        content = (raw or "").strip()
        if not content:
            return []
        # Strip markdown code blocks if present (e.g. ```json ... ```)
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.lstrip().lower().startswith("json"):
                content = content.lstrip()[4:]
            content = content.strip()
        # Strip BOM if present
        if content.startswith("\ufeff"):
            content = content[1:]
        if not content:
            return []
        data = json.loads(content)
        raw_entities = data.get("entities")
        if not isinstance(raw_entities, list):
            return []
        result: list[ResolvedEntity] = []
        iso_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
        for item in raw_entities:
            if not isinstance(item, dict):
                continue
            ticker_raw = item.get("company_ticker")
            ticker = (ticker_raw.strip().upper() if isinstance(ticker_raw, str) and ticker_raw else None) or None
            if not ticker or ticker not in valid_tickers:
                if ticker:
                    logger.debug("Entity resolution LLM returned unknown ticker %s, skipping", ticker)
                continue
            date_expr = item.get("date_expression")
            date_expr = (date_expr.strip() if isinstance(date_expr, str) and date_expr else None) or None
            # Parse date_range if present
            dr = item.get("date_range")
            dr_start: str | None = None
            dr_end: str | None = None
            if isinstance(dr, dict):
                s = dr.get("start", "")
                e = dr.get("end", "")
                if isinstance(s, str) and isinstance(e, str) and iso_pattern.fullmatch(s.strip()) and iso_pattern.fullmatch(e.strip()):
                    dr_start, dr_end = s.strip(), e.strip()
            result.append((ticker, date_expr, dr_start, dr_end))
        logger.info("[debug] entity resolution: query=%r → raw=%s → result=%s", query, raw_entities, result)
        return result
    except (json.JSONDecodeError, KeyError, IndexError, AttributeError) as e:
        raw_content = (raw or "").strip()[:500]
        logger.warning(
            "Entity resolution LLM parse failed: %s (raw: %r)",
            e,
            raw_content,
        )
        return []
    except Exception as e:
        logger.warning("Entity resolution LLM call failed: %s", e)
        return []


def _scope_cache_key(query: str, companies: list[dict[str, str]], today_iso: str, conversation_context: list[str] | None = None) -> tuple:
    """Build cache key for entity resolution scope. Companies key is stable for same ticker/name set."""
    normalized = query.strip().lower()[:500]
    companies_key = tuple(sorted((c.get("ticker", ""), c.get("name", "")) for c in companies))
    context_key = tuple((conversation_context or [])[-4:])
    return (normalized, today_iso, companies_key, context_key)


async def resolve_company_and_date(
    query: str,
    companies: list[dict[str, str]],
    today_iso: str,
    conversation_context: list[str] | None = None,
) -> SimpleRAGScope:
    """Resolve scope from LLM list of (entity, date) pairs.

    Calls LLM to extract (company_ticker, date_expression) from the query.
    Multi-entity: resolve to (ticker, call_date) pairs for OR filter; if none found, use no filter.
    Results are cached (24h TTL) to reduce latency on repeated/similar queries.
    """
    cache_key = _scope_cache_key(query, companies, today_iso, conversation_context)
    if cache_key in _SCOPE_CACHE:
        return SimpleRAGScope(resolved_date_pairs=_SCOPE_CACHE[cache_key])

    entities: list[ResolvedEntity] = []
    if companies:
        resolution_query = query
        if conversation_context:
            prior = "; ".join(conversation_context)  # all 4 prior queries
            resolution_query = f"[Prior queries for context: {prior}] Current query: {query}"
        entities = await _resolve_entities_via_llm(resolution_query, companies, today_iso)

    if not entities:
        _SCOPE_CACHE[cache_key] = None
        return SimpleRAGScope()

    # Separate range entities (multi-quarter windows) from point entities (single call)
    range_entities = [(ticker, dr_start, dr_end) for ticker, _, dr_start, dr_end in entities if dr_start and dr_end]
    point_entities = [
        (ticker, _date_expression_to_iso(date_expr, today_iso) or today_iso)
        for ticker, date_expr, dr_start, _ in entities
        if not dr_start
    ]

    resolved_pairs: list[tuple[str, str]] = []

    coros = []
    if range_entities:
        coros.append(resolve_entities_to_call_dates_in_ranges(range_entities))
    if point_entities:
        coros.append(resolve_entities_to_call_dates(point_entities))

    t0 = time.perf_counter()
    results = await asyncio.gather(*coros)
    logger.info("[latency] resolve call dates (parallel): %.3fs", time.perf_counter() - t0)
    for r in results:
        resolved_pairs.extend(r)

    if not resolved_pairs:
        logger.warning("Entity resolution: no call dates found for %s, using no filter", entities)

    logger.info("Simple RAG: %d entities → %d resolved pairs", len(entities), len(resolved_pairs))
    pairs = resolved_pairs or None
    _SCOPE_CACHE[cache_key] = pairs
    return SimpleRAGScope(resolved_date_pairs=pairs)


def _format_context_for_prompt(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks for the prompt; include company_ticker and call_date so the LLM knows which company/period each chunk is from."""
    parts = []
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata") or {}
        ticker = meta.get("company_ticker", "")
        title = meta.get("title", "Unknown")
        call_date = meta.get("call_date", "")
        sim = c.get("similarity", 0.0)
        # company_ticker comes from ingestion (required at upload); title may only have date
        label_parts = [p for p in [ticker, title, call_date] if p]
        source_label = " | ".join(label_parts) if label_parts else "Unknown"
        parts.append(f"[Source {i}: {source_label} (relevance: {sim:.2f})]\n{c.get('content', '')}")
    return "\n\n---\n\n".join(parts) if parts else ""
