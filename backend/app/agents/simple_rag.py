"""Simple RAG: single retrieval + one LLM call (no agentic tool loop).

Entity resolution maps natural language → (tickers, temporal scope) via a cheap LLM call.
Temporal resolution then maps the temporal intent to specific (ticker, call_date) pairs
using a deterministic lookup against cached available periods.
Query rewriting expands the query into 2-3 retrieval variants (actual vs guidance,
metric aliases) so retrieval covers different interpretations in one pass.
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import date
from typing import Any

import tiktoken
from cachetools import TTLCache

from app.agents.prompt_utils import _sanitize_for_prompt
from app.config import get_settings
from app.rag.embeddings import get_openai_client
from app.rag.fiscal_calendar import (
    compute_cy_quarter_end,
    cy_quarter_label_from_period_end,
    period_end_to_label,
)

_tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

logger = logging.getLogger(__name__)

# Scope resolution cache (24h TTL). Key: (normalized_query, companies_key, context_key).
_SCOPE_CACHE: TTLCache = TTLCache(maxsize=500, ttl=86400)


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
class TemporalIntent:
    """What the user meant temporally, as extracted by the LLM."""
    type: str = "unspecified"  # "latest" | "specific_quarter" | "range" | "unspecified"
    quarter: int | None = None         # 1-4 (for specific_quarter)
    year: int | None = None            # e.g. 2025 (for specific_quarter)
    # Range fields — anchored or rolling:
    num_quarters: int | None = None    # rolling window ("last 8 quarters")
    start_year: int | None = None      # anchor start year
    start_quarter: int | None = None   # anchor start quarter (default Q1)
    end_year: int | None = None        # anchor end year
    end_quarter: int | None = None     # anchor end quarter (default Q4)


@dataclass
class SimpleRAGScope:
    """Resolved scope for retrieval."""
    tickers: list[str] | None = None
    # Resolved (ticker, call_date) pairs for temporal filtering. When set,
    # retrieval is scoped to these exact pairs (more specific than tickers alone).
    ticker_date_pairs: list[tuple[str, str]] | None = None
    temporal_intent: TemporalIntent | None = None


def _scope_cache_key(
    query: str,
    companies: list[dict[str, str]],
    conversation_context: list[str] | None = None,
) -> tuple:
    normalized = query.strip().lower()[:500]
    companies_key = tuple(sorted((c.get("ticker", ""), c.get("name", "")) for c in companies))
    context_key = tuple((conversation_context or [])[-5:])
    return (normalized, companies_key, context_key)


def _strip_json_fences(text: str) -> str:
    """Remove markdown code fences from LLM JSON output."""
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
        text = text.strip()
    return text


# Matches a 4-digit year (2019-2030) that is NOT preceded by "Q1"–"Q4"
_BARE_YEAR_RE = re.compile(
    r"(?<![Qq]\d\s)(?<![Qq])\b(20[1-3]\d)\b"
)
# Matches "Q1 2024" style references — if present, the year is NOT bare
_QUARTER_YEAR_RE = re.compile(r"[Qq][1-4]\s*(?:FY|CY)?\s*\d{2,4}", re.IGNORECASE)
# "last year" / "past year" / … → rolling 4 calendar quarters (see _fix_last_year)
_LAST_YEAR_PHRASE_RE = re.compile(
    r"\b(?:"
    r"(?:last|past|previous)\s+year|"
    r"over\s+the\s+last\s+year|"
    r"in\s+the\s+last\s+year|"
    r"during\s+the\s+last\s+year"
    r")\b",
    re.IGNORECASE,
)


def _fix_bare_year(query: str, temporal: TemporalIntent) -> TemporalIntent:
    """Override LLM temporal classification when the query contains a bare year.

    If the user wrote "Meta 2024 revenue" the LLM may return "latest" or
    "specific_quarter".  This function detects a year without an adjacent
    quarter reference and forces the intent to an anchored range covering
    all 4 quarters of that year.
    """
    if temporal.type == "range" and temporal.start_year is not None:
        return temporal  # already correct

    # Skip if the query has an explicit quarter reference like "Q3 2024"
    if _QUARTER_YEAR_RE.search(query):
        return temporal

    m = _BARE_YEAR_RE.search(query)
    if not m:
        return temporal

    year = int(m.group(1))
    logger.info("[debug] _fix_bare_year: overriding %s → range(start_year=%d, end_year=%d)", temporal.type, year, year)
    temporal.type = "range"
    temporal.start_year = year
    temporal.end_year = year
    temporal.start_quarter = None
    temporal.end_quarter = None
    temporal.num_quarters = None
    return temporal


def _fix_last_year(
    query: str,
    conversation_context: list[str] | None,
    temporal: TemporalIntent,
) -> TemporalIntent:
    """Map 'last year' / 'past year' / … to rolling 4 calendar quarters from today.

    Skips when the current query names an explicit calendar year (bare year) or
    quarter+year, so 'revenue in 2024' stays an anchored full-year range.
    """
    if _BARE_YEAR_RE.search(query) or _QUARTER_YEAR_RE.search(query):
        return temporal

    combined = " ".join([*(conversation_context or []), query])
    if not _LAST_YEAR_PHRASE_RE.search(combined):
        return temporal

    logger.info(
        "[debug] _fix_last_year: overriding %s → range(num_quarters=4, rolling CY)",
        temporal.type,
    )
    temporal.type = "range"
    temporal.num_quarters = 4
    temporal.start_year = None
    temporal.end_year = None
    temporal.start_quarter = None
    temporal.end_quarter = None
    temporal.quarter = None
    temporal.year = None
    return temporal


async def _resolve_scope_via_llm(
    query: str,
    companies: list[dict[str, str]],
    conversation_context: list[str] | None = None,
) -> tuple[list[str], TemporalIntent]:
    """Extract company tickers AND temporal scope from query using a cheap LLM call.

    Returns (tickers, temporal_intent).
    """
    if not companies:
        return [], TemporalIntent()

    valid_tickers = {c.get("ticker", "").upper() for c in companies if c.get("ticker")}
    company_list = ", ".join(
        f"{c.get('ticker', '').upper()}: {c.get('name', '')}" for c in companies if c.get("ticker")
    )
    sanitized_query = _sanitize_for_prompt(query, max_len=500)
    resolution_query = sanitized_query
    if conversation_context:
        prior = "; ".join(_sanitize_for_prompt(c, max_len=300) for c in conversation_context)
        resolution_query = f"[Prior context: {prior}] Current query: {sanitized_query}"

    prompt = f"""Available companies (ticker: name): {company_list}

From the user query and conversation context, extract:
1. Company tickers being asked about
2. Temporal scope (which time period)

TICKERS:
- Use conversation context: follow-ups reference prior companies ("what about their Q2?" → prior company)
- Fuzzy match: "ServieNow" → NOW, "amazon" → AMZN
- If ambiguous with multiple prior companies, default to the most recently mentioned
- Return [] only if genuinely no company is identifiable

TEMPORAL SCOPE — classify as one of:
- "latest": user wants the most recent quarter. This is the DEFAULT when no time period is mentioned but a financial metric is asked AND no prior conversation established a specific time period (e.g. "Samsara revenue" with no prior context → latest).
- "specific_quarter": user references a specific quarter, e.g. "Q1 25", "Q4 FY2025", "Q3 2024". MUST have a quarter number.
- "range": user wants data across multiple quarters. Use optional anchor fields to specify the window:
    - IMPORTANT: A bare year WITHOUT a quarter (e.g. "Meta 2024 revenue", "Amazon capex for 2023", "revenue in 2024") is a FULL YEAR query — use range with start_year and end_year set to that year. Do NOT use "latest" or "specific_quarter" for these.
    - If a specific year is mentioned (e.g. "in 2024", "for 2023"): set start_year and end_year to that year.
    - If a year span is mentioned (e.g. "from 2021 to 2023"): set start_year and end_year.
    - If a specific quarter start/end is mentioned (e.g. "from Q2 2022 to Q4 2023"): set start_year, start_quarter, end_year, end_quarter.
    - If a rolling window from today (e.g. "last 8 quarters", "revenue trend"): set only num_quarters.
    - Phrases "last year", "past year", "previous year", "over the last year" mean the LAST 4 CALENDAR QUARTERS rolling back from today — use range with ONLY num_quarters=4 (no start_year/end_year). Same as "last 4 quarters" in intent.
    - start_quarter defaults to Q1, end_quarter defaults to Q4 when omitted.
- "unspecified": genuinely no temporal reference AND not a specific metrics question (e.g. "tell me about Samsara", "what companies are available")

TEMPORAL CARRYOVER: If the user's follow-up asks about a financial metric without specifying a time period, AND the prior conversation established a specific time period (e.g. "Q4 25"), carry that time period forward — but ONLY if the follow-up is about the SAME company. When the user switches to a DIFFERENT company, do NOT carry forward the prior time period — use "latest" instead. Examples:
- Prior: "ServiceNow Q4 25 revenue" → Follow-up: "what about their margins?" → SAME company, carry forward specific_quarter Q4 2025
- Prior: "Samsara last 8 quarters revenue" → Follow-up: "and their gross margin?" → SAME company, carry forward range num_quarters=8
- Prior: "tell me about Samsara" → Follow-up: "what's their revenue?" → no prior time period, use "latest"
- Prior: "Samsara Q1 25 revenue" → Follow-up: "ServiceNow revenue" → DIFFERENT company, do NOT carry forward, use "latest"
- Prior: "Amazon capex for 2024" → Follow-up: "what about their revenue?" → SAME company, carry forward range start_year=2024, end_year=2024

For specific_quarter:
- Extract quarter (1-4) and year (4-digit)
- Always interpret as calendar year — "Q1 25", "Q1 FY25", "Q1 CY25" all mean CY Q1 2025 (Jan–Mar 2025)
For range:
- Anchored: set start_year/end_year (and optionally start_quarter/end_quarter)
- Rolling: set num_quarters only (counts back from today)

Reply with only JSON:
{{"tickers": ["TICKER1"], "temporal": {{"type": "latest"}}}}
{{"tickers": ["NOW", "CRM"], "temporal": {{"type": "specific_quarter", "quarter": 4, "year": 2025}}}}
{{"tickers": ["NOW"], "temporal": {{"type": "range", "num_quarters": 8}}}}
{{"tickers": ["AMZN"], "temporal": {{"type": "range", "start_year": 2024, "end_year": 2024}}}}
{{"tickers": ["AMZN"], "temporal": {{"type": "range", "start_year": 2021, "end_year": 2023}}}}
{{"tickers": ["META"], "temporal": {{"type": "range", "start_quarter": 2, "start_year": 2022, "end_quarter": 4, "end_year": 2023}}}}

User query: {resolution_query}"""

    try:
        client = get_openai_client()
        t0 = time.perf_counter()
        response = await client.responses.create(
            model=get_settings().openai_model,
            input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            max_output_tokens=150,
        )
        logger.info("[latency] _resolve_scope_via_llm: %.3fs", time.perf_counter() - t0)
        raw = (
            response.output_text
            if hasattr(response, "output_text")
            else _extract_text_from_responses_output(getattr(response, "output", []))
        )
        content = (raw or "").strip()
        if not content:
            return [], TemporalIntent()

        content = _strip_json_fences(content)
        data = json.loads(content)

        # Parse tickers
        raw_tickers = data.get("tickers", [])
        tickers = [t.upper() for t in raw_tickers if isinstance(t, str) and t.strip().upper() in valid_tickers]

        # Parse temporal intent
        temporal_data = data.get("temporal", {})
        temporal = TemporalIntent(
            type=temporal_data.get("type", "unspecified"),
            quarter=temporal_data.get("quarter"),
            year=temporal_data.get("year"),
            num_quarters=temporal_data.get("num_quarters"),
            start_year=temporal_data.get("start_year"),
            start_quarter=temporal_data.get("start_quarter"),
            end_year=temporal_data.get("end_year"),
            end_quarter=temporal_data.get("end_quarter"),
        )
        # Normalize 2-digit years
        for attr in ("year", "start_year", "end_year"):
            v = getattr(temporal, attr)
            if v and v < 100:
                setattr(temporal, attr, v + 2000)

        # Code-level override: if query contains a bare year (no quarter), force range
        temporal = _fix_bare_year(query, temporal)
        temporal = _fix_last_year(query, conversation_context, temporal)

        logger.info("[debug] scope resolution: query=%r → tickers=%s, temporal=%s", query, tickers, temporal)
        return tickers, temporal
    except Exception as e:
        logger.warning("Scope resolution failed: %s", e)
        return [], TemporalIntent()


def _find_closest_period(
    periods: list[dict[str, Any]],
    target_end: date,
) -> dict[str, Any] | None:
    """Find the period whose period_end is closest to target_end."""
    if not periods:
        return None
    return min(
        periods,
        key=lambda p: abs((date.fromisoformat(p["period_end"]) - target_end).days),
    )


def _resolve_temporal(
    tickers: list[str],
    temporal: TemporalIntent,
    available_periods: dict[str, list[dict[str, Any]]],
    today: date | None = None,
) -> list[tuple[str, str]] | None:
    """Map temporal intent + tickers to specific (ticker, call_date) pairs.

    Returns None when temporal resolution doesn't apply (unspecified),
    meaning retrieval should use ticker-only filtering.
    """
    today = today or date.today()

    if temporal.type == "unspecified":
        return None

    if temporal.type == "latest":
        pairs = []
        for t in tickers:
            ticker_periods = available_periods.get(t, [])
            if ticker_periods:
                pairs.append((t, ticker_periods[0]["call_date"]))  # sorted desc
        if pairs:
            logger.info("[debug] temporal resolution: latest → (ticker, call_date): %s", pairs)
        return pairs or None

    if temporal.type == "specific_quarter":
        if temporal.quarter is None:
            return None
        year = temporal.year or today.year

        # Always resolve as calendar year quarter
        target_end = compute_cy_quarter_end(temporal.quarter, year)
        logger.info(
            "[debug] temporal resolution: specific_quarter CY target: %s",
            cy_quarter_label_from_period_end(target_end),
        )

        pairs = []
        for t in tickers:
            ticker_periods = available_periods.get(t, [])
            if not ticker_periods:
                continue
            closest = _find_closest_period(ticker_periods, target_end)
            if closest:
                pairs.append((t, closest["call_date"]))

        return pairs or None

    if temporal.type == "range":
        # Unified range: anchored (start/end year+quarter) or rolling (num_quarters).
        is_anchored = temporal.start_year is not None or temporal.end_year is not None

        if is_anchored:
            # Anchored range — enumerate CY quarters between start and end
            s_year = temporal.start_year or temporal.end_year or today.year
            e_year = temporal.end_year or temporal.start_year or today.year
            s_quarter = temporal.start_quarter or 1
            e_quarter = temporal.end_quarter or 4

            cy_targets: list[date] = []
            cur_q, cur_y = s_quarter, s_year
            while (cur_y, cur_q) <= (e_year, e_quarter):
                cy_targets.append(compute_cy_quarter_end(cur_q, cur_y))
                cur_q += 1
                if cur_q > 4:
                    cur_q = 1
                    cur_y += 1
        else:
            # Rolling range — count back from most recently completed CY quarter
            n = temporal.num_quarters or 4
            cy_targets: list[date] = []
            cur_q = (today.month - 1) // 3 + 1
            cur_y = today.year
            if compute_cy_quarter_end(cur_q, cur_y) > today:
                cur_q -= 1
                if cur_q < 1:
                    cur_q = 4
                    cur_y -= 1
            for _ in range(n):
                cy_targets.append(compute_cy_quarter_end(cur_q, cur_y))
                cur_q -= 1
                if cur_q < 1:
                    cur_q = 4
                    cur_y -= 1

        cy_labels = [cy_quarter_label_from_period_end(d) for d in cy_targets]
        logger.info(
            "[debug] temporal resolution: range (%s) CY quarter targets: %s",
            "anchored" if is_anchored else "rolling",
            cy_labels,
        )

        pairs: list[tuple[str, str, str]] = []  # (ticker, call_date, period_end)
        for t in tickers:
            ticker_periods = available_periods.get(t, [])
            if not ticker_periods:
                continue
            seen_call_dates: set[str] = set()
            for target_end in cy_targets:
                closest = _find_closest_period(ticker_periods, target_end)
                if closest and closest["call_date"] not in seen_call_dates:
                    pairs.append((t, closest["call_date"], closest.get("period_end", "")))
                    seen_call_dates.add(closest["call_date"])
        pairs.sort(key=lambda p: p[2], reverse=True)
        resolved_pairs = [(t, cd) for t, cd, _ in pairs]
        if resolved_pairs:
            logger.info(
                "[debug] temporal resolution: range → resolved (ticker, call_date): %s",
                resolved_pairs,
            )
        return resolved_pairs or None

    return None


_SKIP_REWRITE_RE = re.compile(
    r"^\s*(?:sort|order|show|display|list|format|group|arrange|compare|rank|table|chart|give me|what is|what are|who is)\b",
    re.IGNORECASE,
)


def _should_skip_rewrite(query: str) -> bool:
    """Return True when query rewriting adds no retrieval value.

    Skips the LLM call for:
    - Short queries (≤ 5 words): single-concept, no aliases to expand.
    - Display/formatting instructions: the LLM would return the original anyway.
    """
    words = query.strip().split()
    if len(words) <= 5:
        return True
    if _SKIP_REWRITE_RE.match(query.strip()):
        return True
    return False


async def _rewrite_query_for_retrieval(query: str) -> list[str]:
    """Expand a user query into 2-3 retrieval queries covering different interpretations.

    For financial queries this covers: reported/actual results vs guidance/outlook,
    and alternative metric names. Falls back to [query] on any error.
    """
    sanitized_query = _sanitize_for_prompt(query, max_len=500)
    prompt = f"""You are preparing search queries for an earnings call transcript database.

Given this user query: "{sanitized_query}"

Generate 2-3 short, distinct retrieval queries that together cover different interpretations:
- Reported/actual results AND forward guidance or outlook (if about a financial metric)
- Alternative names for the same metric (e.g. "revenue" → also "total revenue")
- If the query is purely a formatting or display instruction ("sort by date", "show as table", "chronological order"), return just the original query unchanged.

Keep each query under 15 words. Do not add company names — those are handled separately.

Reply with only JSON: {{"queries": ["...", "...", "..."]}}"""

    try:
        client = get_openai_client()
        t0 = time.perf_counter()
        response = await client.responses.create(
            model=get_settings().openai_model,
            input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            max_output_tokens=150,
        )
        logger.info("[latency] _rewrite_query_for_retrieval: %.3fs", time.perf_counter() - t0)
        raw = (
            response.output_text
            if hasattr(response, "output_text")
            else _extract_text_from_responses_output(getattr(response, "output", []))
        )
        content = (raw or "").strip()
        if not content:
            return [query]
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.lstrip().lower().startswith("json"):
                content = content.lstrip()[4:]
            content = content.strip()
        data = json.loads(content)
        queries = [q for q in data.get("queries", []) if isinstance(q, str) and q.strip()]
        return queries or [query]
    except Exception as e:
        logger.warning("Query rewriting failed: %s", e)
        return [query]


async def resolve_company_and_date(
    query: str,
    companies: list[dict[str, str]],
    today_iso: str,
    conversation_context: list[str] | None = None,
    available_periods: dict[str, list[dict[str, Any]]] | None = None,
) -> SimpleRAGScope:
    """Resolve company tickers and temporal scope from the query.

    When available_periods is provided, also performs temporal resolution
    to map the user's temporal intent to specific (ticker, call_date) pairs.
    Results cached 24h.
    """
    cache_key = _scope_cache_key(query, companies, conversation_context)
    if cache_key in _SCOPE_CACHE:
        return _SCOPE_CACHE[cache_key]

    tickers, temporal = (
        await _resolve_scope_via_llm(query, companies, conversation_context)
        if companies else ([], TemporalIntent())
    )

    scope = SimpleRAGScope(
        tickers=tickers or None,
        temporal_intent=temporal,
    )

    # Temporal resolution: map intent → specific (ticker, call_date) pairs
    if tickers and available_periods and temporal.type != "unspecified":
        today = date.fromisoformat(today_iso) if today_iso else date.today()
        pairs = _resolve_temporal(tickers, temporal, available_periods, today)
        if pairs:
            scope.ticker_date_pairs = pairs
            logger.info("[debug] temporal resolution: %s → %d pairs", temporal.type, len(pairs))

    _SCOPE_CACHE[cache_key] = scope
    return scope


def trim_chunks_to_token_budget(
    chunks: list[dict[str, Any]],
    budget: int | None = None,
) -> list[dict[str, Any]]:
    """Drop lowest-relevance chunks until total content fits within *budget* tokens.

    Chunks are assumed to be sorted by relevance (highest first). Financial summary
    chunks (chunk_type == "financials") are prioritised and only dropped last.
    """
    budget = budget or get_settings().context_token_budget
    # Partition: financials chunks are high-value, keep them separate.
    financials: list[dict[str, Any]] = []
    regular: list[dict[str, Any]] = []
    for c in chunks:
        meta = c.get("metadata") or {}
        if meta.get("chunk_type") == "financials":
            financials.append(c)
        else:
            regular.append(c)

    # Start with all financials + regular (already relevance-sorted), trim from the tail.
    ordered = financials + regular
    total = sum(len(_tokenizer.encode(c.get("content", ""))) for c in ordered)

    if total <= budget:
        return ordered  # financials-first order, consistent with the trim path

    # Drop lowest-relevance regular chunks first, then financials if still over.
    trimmed = list(ordered)
    while total > budget and trimmed:
        dropped = trimmed.pop()
        total -= len(_tokenizer.encode(dropped.get("content", "")))

    logger.info(
        "[token_budget] trimmed %d → %d chunks (~%d tokens, budget=%d)",
        len(chunks), len(trimmed), total, budget,
    )
    return trimmed


def reorder_chunks_for_range(
    chunks: list[dict[str, Any]],
    scope: "SimpleRAGScope",
) -> list[dict[str, Any]]:
    """For range queries, reorder chunks reverse-chronologically with financials first per period.

    Instead of relevance-sorted order (which mixes periods), this groups chunks by
    period and sorts most-recent-first so the LLM presents data in reverse
    chronological order without confusing figures across periods.
    """
    if not scope.temporal_intent or scope.temporal_intent.type != "range":
        return chunks

    def _sort_key(chunk: dict[str, Any]) -> tuple[str, int]:
        meta = chunk.get("metadata") or {}
        period_end = meta.get("period_end", "")
        call_date = meta.get("call_date", "")
        # Invert date string for descending sort while keeping financials first
        date_key = period_end or call_date or "0000"
        inverted = "".join(chr(ord("9") - ord(c)) if c.isdigit() else c for c in date_key)
        is_financials = 0 if meta.get("chunk_type") == "financials" else 1
        return (inverted, is_financials)

    return sorted(chunks, key=_sort_key)


def build_resolution_note(
    scope: "SimpleRAGScope",
    available_periods: dict[str, list[dict[str, Any]]],
    query: str = "",
) -> str:
    """Build a note explaining temporal resolution so the answer LLM knows
    which periods were matched, even when fiscal labels differ from the user's wording."""
    if not scope.ticker_date_pairs or not scope.temporal_intent:
        return ""

    intent = scope.temporal_intent
    if intent.type == "unspecified":
        return ""

    # Describe the target period
    if intent.type == "latest":
        target_desc = "the most recent available quarter"
    elif intent.type == "specific_quarter" and intent.quarter:
        year = intent.year or "current year"
        target_desc = f"CY Q{intent.quarter} {year} (calendar year)"
    elif intent.type == "range":
        if intent.start_year and intent.end_year and intent.start_year == intent.end_year:
            sq = intent.start_quarter or 1
            eq = intent.end_quarter or 4
            if sq == 1 and eq == 4:
                target_desc = f"all 4 quarters of calendar year {intent.start_year}"
            else:
                target_desc = f"CY Q{sq}–Q{eq} {intent.start_year}"
        elif intent.start_year or intent.end_year:
            s = f"Q{intent.start_quarter or 1} {intent.start_year or intent.end_year}"
            e = f"Q{intent.end_quarter or 4} {intent.end_year or intent.start_year}"
            target_desc = f"all quarters from CY {s} through CY {e}"
        else:
            n = intent.num_quarters or 4
            target_desc = f"the last {n} calendar quarters (rolling from today)"
    else:
        target_desc = "the requested time period"

    lines = [
        f"TEMPORAL RESOLUTION: The user's query has been resolved to {target_desc}. "
        "The context below contains the CORRECT data for this request. Resolved periods:"
    ]

    for ticker, call_date in scope.ticker_date_pairs:
        ticker_periods = available_periods.get(ticker, [])
        match = next((p for p in ticker_periods if p["call_date"] == call_date), None)
        if match:
            fq = match.get("fiscal_quarter", "")
            pe_str = match.get("period_end", "")
            label = ""
            if pe_str:
                try:
                    label = period_end_to_label(date.fromisoformat(pe_str))
                except (ValueError, TypeError):
                    pass
            parts = [ticker]
            if fq:
                parts.append(fq)
            if label:
                parts.append(f"covers {label}")
            lines.append(f"  - {' | '.join(parts)}")
        else:
            lines.append(f"  - {ticker} | call date {call_date}")

    # Build explicit user-query → fiscal-quarter mapping
    mapping_lines = []
    for ticker, call_date in scope.ticker_date_pairs:
        ticker_periods = available_periods.get(ticker, [])
        match = next((p for p in ticker_periods if p["call_date"] == call_date), None)
        if match and match.get("fiscal_quarter"):
            fq = match["fiscal_quarter"]
            pe_str = match.get("period_end", "")
            label = ""
            if pe_str:
                try:
                    label = f" ({period_end_to_label(date.fromisoformat(pe_str))})"
                except (ValueError, TypeError):
                    pass
            mapping_lines.append(f"  → {ticker}: the user's query = {fq}{label} in the transcript")

    if mapping_lines:
        lines.append("MAPPING (user's query → transcript fiscal quarter):")
        lines.extend(mapping_lines)

    lines.append(
        "IMPORTANT: Do NOT second-guess this resolution or say data is missing. "
        "The resolved fiscal quarter IS the correct answer to the user's query — report it directly. "
        "MANDATORY: Always format fiscal quarters with the calendar period in parentheses — "
        "e.g. \"Q4 FY2025 (Apr–Jun 2025) revenue was $70.1B\". "
        "Never write a bare fiscal quarter without its calendar period."
    )

    # Multi-period instructions: help LLM organize multi-period data correctly
    if intent.type == "range":
        is_full_year = (
            intent.start_year is not None
            and intent.start_year == intent.end_year
            and (intent.start_quarter or 1) == 1
            and (intent.end_quarter or 4) == 4
        )

        lines.append("")
        lines.append(
            "RANGE QUERY INSTRUCTIONS: The context below is organized in reverse chronological order "
            "(most recent first). "
            "Each period's FINANCIAL SUMMARY chunk contains the authoritative figures for that quarter. "
            "You MUST:"
        )

        if is_full_year:
            year = intent.start_year
            # Check if user explicitly asked for guidance/outlook
            q_lower = query.lower()
            wants_guidance = any(w in q_lower for w in ("guidance", "outlook", "forecast", "guide"))
            figure_type = "GUIDED/OUTLOOK" if wants_guidance else "REPORTED/ACTUAL"
            label_match = "Guided" if wants_guidance else "Reported"

            lines.append(
                f"  *** FULL-YEAR QUERY ({year}) ***\n"
                f"  The user wants the TOTAL annual {figure_type} figure for {year}."
            )
            lines.append(
                f"  STEP 1: Look in the FINANCIAL SUMMARY chunks for a {label_match} annual/full-year "
                f"figure for {year}. If found, use it directly.\n"
                f"  STEP 2 (if no annual figure): Sum the 4 quarterly {label_match} values from each "
                "quarter's FINANCIAL SUMMARY. Present as:\n"
                f"    Q1: $A + Q2: $B + Q3: $C + Q4: $D = $TOTAL for {year}\n"
                "  MANDATORY: Show per-quarter breakdown. NEVER return just one quarter's figure.\n"
                f"  MANDATORY: Only use lines marked '{label_match}' in the FINANCIAL SUMMARY chunks."
            )
        else:
            lines.append(
                "  1. Present data in REVERSE CHRONOLOGICAL ORDER (most recent quarter first). "
                "Use a structured table or clearly labeled list with one entry per period. "
                "Each entry MUST include the fiscal quarter with calendar period, "
                "e.g. \"Q4 FY2025 (Oct–Dec 2025)\"."
            )
            lines.append(
                "  2. Match each figure to its EXACT fiscal quarter — never combine or confuse "
                "figures from different periods. A YTD or trailing-12-month figure belongs to "
                "the quarter it was reported in, not to prior quarters."
            )
            lines.append(
                "  3. If the user asks for an annual total but only quarterly figures are available, "
                "sum the individual quarters and show the calculation. Do NOT use a YTD figure "
                "from one quarter as the annual total for a different fiscal year."
            )

        lines.append(
            "  Use only REPORTED/ACTUAL figures — never guidance or outlook — "
            "unless the user explicitly asks for guidance. "
            "If a specific figure is not available for a period, say 'Not reported' "
            "rather than approximating from other periods."
        )

    return "\n".join(lines)


def rewrite_query_for_answer_llm(
    query: str,
    scope: "SimpleRAGScope",
    available_periods: dict[str, list[dict[str, Any]]],
) -> str:
    """Rewrite the user query replacing calendar quarter references with the
    company's fiscal quarter label so the answer LLM never sees conflicting
    quarter numbers between the query and the transcript.

    Example: "MSFT Q2 25 revenue" → "MSFT Q4 FY2025 (Apr–Jun 2025) revenue"
    """
    if not scope.ticker_date_pairs or not scope.temporal_intent:
        return query
    intent = scope.temporal_intent
    if intent.type not in ("specific_quarter", "latest"):
        return query

    # Build ticker → fiscal quarter label mapping
    ticker_labels: dict[str, str] = {}
    for ticker, call_date in scope.ticker_date_pairs:
        ticker_periods = available_periods.get(ticker, [])
        match = next((p for p in ticker_periods if p["call_date"] == call_date), None)
        if match and match.get("fiscal_quarter"):
            fq = match["fiscal_quarter"]
            pe_str = match.get("period_end", "")
            label = ""
            if pe_str:
                try:
                    label = f" ({period_end_to_label(date.fromisoformat(pe_str))})"
                except (ValueError, TypeError):
                    pass
            ticker_labels[ticker] = f"{fq}{label}"

    if not ticker_labels:
        return query

    # Append fiscal quarter context to the query
    if len(ticker_labels) == 1:
        ticker, fq_label = next(iter(ticker_labels.items()))
        return f"{query} [{ticker} {fq_label}]"
    else:
        parts = [f"{t} {l}" for t, l in ticker_labels.items()]
        return f"{query} [{'; '.join(parts)}]"


def _format_context_for_prompt(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks for the prompt; include company_ticker, call_date, and calendar period so the LLM knows which company/period each chunk is from."""
    parts = []
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata") or {}
        ticker = meta.get("company_ticker", "")
        company_name = meta.get("company_name", "")
        call_date = meta.get("call_date", "")
        fiscal_quarter = meta.get("fiscal_quarter", "")
        period_end_str = meta.get("period_end", "")
        sim = c.get("similarity", 0.0)
        company_label = f"{ticker} ({company_name})" if company_name else ticker

        # Build period label: fiscal quarter + calendar period (from period_end) + call date
        period_parts = []
        if fiscal_quarter:
            period_parts.append(fiscal_quarter)
        if period_end_str:
            try:
                pe = date.fromisoformat(period_end_str)
                period_parts.append(period_end_to_label(pe))
            except (ValueError, TypeError):
                pass
        if call_date:
            period_parts.append(call_date)
        period = " | ".join(period_parts)

        source_label = f"{company_label} | {period}" if period else company_label or "Unknown"
        parts.append(f"[Source {i}: {source_label} (relevance: {sim:.2f})]\n{c.get('content', '')}")
    return "\n\n---\n\n".join(parts) if parts else ""
