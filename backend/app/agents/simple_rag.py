"""Simple RAG: single retrieval + one LLM call (no agentic tool loop).

Entity (company) and date are resolved via a cheap LLM call returning a list of
(entity, date) pairs. All queries use this path.
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from app.conversations.service import get_recent_user_queries
from app.rag.embeddings import get_openai_client
from app.rag.retriever import resolve_entities_to_call_dates, retrieve_relevant_chunks

logger = logging.getLogger(__name__)

# Quarter end dates (approximate): Q1 -> Mar 31, Q2 -> Jun 30, Q3 -> Sep 30, Q4 -> Dec 31
_QUARTER_END = {"1": "03-31", "2": "06-30", "3": "09-30", "4": "12-31"}

# Entity resolution: small fast model, low max_tokens to minimize latency
_RESOLUTION_MODEL = "gpt-4o-mini"
_RESOLUTION_MAX_TOKENS = 120

# (company_ticker, date_expression) per entity; ticker is validated against known companies
ResolvedEntity = tuple[str, str | None]


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
    """Resolved scope for retrieval and LLM."""

    company_ticker: str | None
    as_of_date: str | None
    # For multi-entity: pre-resolved (ticker, call_date) pairs; enables single retrieval with OR filter.
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
    prompt = f"""Today's date is {today_iso}.

Available companies (ticker: name): {company_list}

From the user query, extract a list of (company, date) pairs the user is asking about.
- Single company / single period: return one object, e.g. [{{ "company_ticker": "ACME", "date_expression": "last_quarter" }}].
- Comparing multiple companies: return one object per company, e.g. [{{ "company_ticker": "ACME", "date_expression": "last_quarter" }}, {{ "company_ticker": "GLOBEX", "date_expression": "last_quarter" }}].
- Same company, multiple time periods (e.g. "change from previous quarter", "vs last quarter", "quarter over quarter"): return multiple objects with the same company_ticker and the two date_expressions, e.g. [{{ "company_ticker": "NOW", "date_expression": "last_quarter" }}, {{ "company_ticker": "NOW", "date_expression": "current_quarter" }}].
- Multiple companies, multiple time periods: return one object per company&time period combo, e.g. [{{ "company_ticker": "ACME", "date_expression": "last_quarter" }}, {{ "company_ticker": "ACME", "date_expression": "current_quarter" }}, {{ "company_ticker": "GLOBEX", "date_expression": "last_quarter" }}, {{ "company_ticker": "GLOBEX", "date_expression": "current_quarter" }}].
- Use only tickers from the list above. date_expression: YYYY-MM-DD, "Q1 2024" style, "last_quarter", "current_quarter", or null for latest.
- If the query is unclear or has no specific company, return an empty list [].

Reply with only a JSON object: {{ "entities": [ {{ "company_ticker": "<ticker>", "date_expression": "<date or null>" }}, ... ] }}

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
            model=_RESOLUTION_MODEL,
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
            result.append((ticker, date_expr))
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


async def resolve_company_and_date(
    query: str,
    request_company_ticker: str | None,
    request_as_of_date: str | None,
    companies: list[dict[str, str]],
    today_iso: str,
) -> SimpleRAGScope:
    """Resolve scope from LLM list of (entity, date) pairs.

    - If both request params are provided, use them (no LLM).
    - Else: call LLM to get (company_ticker, date_expression) pairs.
    - Multi-entity: resolve to (ticker, call_date) pairs for OR filter; if none found, use no filter.
    """
    # Request params: when both provided, use them (e.g. UI sent them)
    if request_company_ticker and request_as_of_date:
        return SimpleRAGScope(
            company_ticker=request_company_ticker,
            as_of_date=request_as_of_date,
        )

    entities: list[ResolvedEntity] = []
    if companies:
        entities = await _resolve_entities_via_llm(query, companies, today_iso)

    if len(entities) >= 2:
        entity_dates = [
            (ticker, _date_expression_to_iso(date_expr, today_iso) or today_iso)
            for ticker, date_expr in entities
        ]
        t0 = time.perf_counter()
        resolved_pairs = await resolve_entities_to_call_dates(entity_dates)
        logger.info("[latency] resolve_entities_to_call_dates: %.3fs", time.perf_counter() - t0)
        if not resolved_pairs:
            logger.warning("Multi-entity: no call dates found for entities %s, using no filter", entity_dates)
        else:
            logger.info("Simple RAG: %d entities -> multi-entity (single retrieval)", len(entities))
        return SimpleRAGScope(
            company_ticker=None,
            as_of_date=None,
            resolved_date_pairs=resolved_pairs if resolved_pairs else None,
        )

    if len(entities) == 1:
        ticker, date_expression = entities[0]
        as_of_date = request_as_of_date
        if date_expression is not None:
            resolved = _date_expression_to_iso(date_expression, today_iso)
            if resolved is not None:
                as_of_date = resolved
        if not as_of_date:
            as_of_date = today_iso
        return SimpleRAGScope(
            company_ticker=ticker,
            as_of_date=as_of_date,
        )

    return SimpleRAGScope(
        company_ticker=request_company_ticker,
        as_of_date=request_as_of_date or today_iso,
    )


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
