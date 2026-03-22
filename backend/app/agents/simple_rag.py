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
import time
from dataclasses import dataclass
from datetime import date
from typing import Any

import tiktoken
from cachetools import TTLCache

from app.agents.prompt_utils import _sanitize_for_prompt
from app.config import get_settings
from app.rag.embeddings import get_openai_client
from app.rag.fiscal_calendar import compute_cy_quarter_end, period_end_to_label

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
    quarter: int | None = None       # 1-4 (for specific_quarter)
    year: int | None = None          # e.g. 2025
    num_quarters: int | None = None  # for range queries


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
- "specific_quarter": user references a specific quarter, e.g. "Q1 25", "Q4 FY2025", "Q3 2024"
- "range": user wants multiple quarters, e.g. "last 8 quarters", "over the past 2 years", "revenue trend"
- "unspecified": genuinely no temporal reference AND not a specific metrics question (e.g. "tell me about Samsara", "what companies are available")

TEMPORAL CARRYOVER: If the user's follow-up asks about a financial metric without specifying a time period, AND the prior conversation established a specific time period (e.g. "Q4 25"), carry that time period forward — do NOT default to "latest". Examples:
- Prior: "ServiceNow Q4 25 revenue" → Follow-up: "what about their margins?" → carry forward specific_quarter Q4 2025
- Prior: "Samsara last 8 quarters revenue" → Follow-up: "and their gross margin?" → carry forward range num_quarters=8
- Prior: "tell me about Samsara" → Follow-up: "what's their revenue?" → no prior time period, use "latest"

For specific_quarter:
- Extract quarter (1-4) and year (4-digit)
- Always interpret as calendar year — "Q1 25", "Q1 FY25", "Q1 CY25" all mean CY Q1 2025 (Jan–Mar 2025)
For range:
- Extract num_quarters (e.g. 8 for "last 8 quarters", 4 for "last year", 20 for "last 5 years")

Reply with only JSON:
{{"tickers": ["TICKER1"], "temporal": {{"type": "latest"}}}}
{{"tickers": ["NOW", "CRM"], "temporal": {{"type": "specific_quarter", "quarter": 4, "year": 2025}}}}
{{"tickers": ["NOW"], "temporal": {{"type": "range", "num_quarters": 8}}}}

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
        )
        # Normalize 2-digit year
        if temporal.year and temporal.year < 100:
            temporal.year += 2000

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
        return pairs or None

    if temporal.type == "specific_quarter":
        if temporal.quarter is None:
            return None
        year = temporal.year or today.year

        # Always resolve as calendar year quarter
        target_end = compute_cy_quarter_end(temporal.quarter, year)

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
        n = temporal.num_quarters or 4
        pairs = []
        for t in tickers:
            ticker_periods = available_periods.get(t, [])
            for p in ticker_periods[:n]:
                pairs.append((t, p["call_date"]))
        return pairs or None

    return None


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
        return chunks  # nothing to trim

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
    """For range queries, reorder chunks chronologically with financials first per period.

    Instead of relevance-sorted order (which mixes periods), this groups chunks by
    period and sorts oldest-first so the LLM can clearly track which data belongs
    to which quarter without confusing figures across periods.
    """
    if not scope.temporal_intent or scope.temporal_intent.type != "range":
        return chunks

    def _sort_key(chunk: dict[str, Any]) -> tuple[str, int]:
        meta = chunk.get("metadata") or {}
        period_end = meta.get("period_end", "")
        call_date = meta.get("call_date", "")
        is_financials = 0 if meta.get("chunk_type") == "financials" else 1
        return (period_end or call_date or "9999", is_financials)

    return sorted(chunks, key=_sort_key)


def build_resolution_note(
    scope: "SimpleRAGScope",
    available_periods: dict[str, list[dict[str, Any]]],
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
        n = intent.num_quarters or 4
        target_desc = f"the last {n} quarters"
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
        "When citing data, use the company's fiscal quarter label from the transcript "
        "and include the calendar period for clarity "
        "(e.g. \"Q4 FY2025 (Apr–Jun 2025) revenue was $70.1B\")."
    )

    # Range-specific instructions: help LLM organize multi-period data correctly
    if intent.type == "range":
        lines.append("")
        lines.append(
            "RANGE QUERY INSTRUCTIONS: The context below is organized chronologically by period. "
            "Each period's FINANCIAL SUMMARY chunk contains the authoritative figures for that quarter. "
            "You MUST:"
        )
        lines.append(
            "  1. Present data as a structured table or clearly labeled list with one entry per period."
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
            "  4. If a specific figure is not available for a period, say 'Not reported' "
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
