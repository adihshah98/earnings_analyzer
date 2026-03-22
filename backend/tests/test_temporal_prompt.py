"""Stress test: Does the answer LLM correctly handle temporal resolution
across various fiscal calendar mismatches?

Scenarios:
1. CRM "Q4 25" → Q4 FY2026 (Nov–Jan) — FY ends Jan, fiscal label differs
2. MSFT "Q2 25" → Q2 FY2025 (Oct–Dec 2024) — FY ends June, big offset
3. NOW vs CRM "Q4 25" — cross-company comparison, different fiscal calendars
4. IOT "latest revenue" — latest quarter, no specific time
5. PANW "last 4 quarters" — range query, FY ends July
"""

import asyncio
import sys
from datetime import date
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_ROOT))

from dotenv import load_dotenv
load_dotenv(BACKEND_ROOT / ".env")

from app.agents.prompt_utils import build_system_prompt, format_known_tickers
from app.agents.simple_rag import (
    SimpleRAGScope,
    TemporalIntent,
    build_resolution_note,
    rewrite_query_for_answer_llm,
    _format_context_for_prompt,
)
from app.prompts.templates import SIMPLE_RAG_SYSTEM_PROMPT
from app.rag.embeddings import get_openai_client
from app.config import get_settings


ALL_COMPANIES = [
    {"ticker": "CRM", "name": "Salesforce"},
    {"ticker": "NOW", "name": "ServiceNow"},
    {"ticker": "MSFT", "name": "Microsoft"},
    {"ticker": "IOT", "name": "Samsara"},
    {"ticker": "PANW", "name": "Palo Alto Networks"},
]


# =============================================================================
# Scenario 1: CRM "Q4 25" → Q4 FY2026 (Nov 2025–Jan 2026)
# User says Q4, transcript says Q4 FY2026. Fiscal label happens to match on
# quarter number but the FY year differs from what user said.
# =============================================================================
SCENARIO_1 = {
    "name": "CRM Q4 25 — FY ends Jan, fiscal year mismatch",
    "query": "CRM Q4 25 revenue",
    "expected": "Should report Q4 FY2026 (Nov 2025–Jan 2026) revenue = $10.03B",
    "available_periods": {
        "CRM": [
            {"call_date": "2026-03-05", "fiscal_quarter": "Q4 FY2026", "period_end": "2026-01-31"},
            {"call_date": "2025-12-03", "fiscal_quarter": "Q3 FY2026", "period_end": "2025-10-31"},
        ],
    },
    "scope": SimpleRAGScope(
        tickers=["CRM"],
        ticker_date_pairs=[("CRM", "2026-03-05")],
        temporal_intent=TemporalIntent(type="specific_quarter", quarter=4, year=2025),
    ),
    "chunks": [
        {
            "chunk_id": "s1-fin",
            "content": (
                "FINANCIAL SUMMARY — Salesforce Q4 FY2026 Earnings Call (2026-03-05)\n\n"
                "REPORTED RESULTS (Q4 FY2026):\n"
                "- Revenue: $10.03B (up 8% YoY)\n"
                "- Non-GAAP Operating Margin: 33.7%\n"
                "- EPS (non-GAAP): $2.63\n"
            ),
            "similarity": 0.92,
            "metadata": {
                "company_ticker": "CRM", "company_name": "Salesforce",
                "call_date": "2026-03-05", "fiscal_quarter": "Q4 FY2026",
                "period_end": "2026-01-31", "chunk_type": "financials",
            },
        },
    ],
}

# =============================================================================
# Scenario 2: MSFT "Q2 25" → Q2 FY2025 (Oct–Dec 2024)
# MSFT FY ends June. CY Q2 2025 = Apr–Jun 2025 (target Jun 30).
# Closest: Q4 FY2025 (Apr–Jun 2025, period_end Jun 30).
# Transcript says "Q4" but user said "Q2". Big mismatch.
# =============================================================================
SCENARIO_2 = {
    "name": "MSFT Q2 25 — FY ends June, quarter number mismatch",
    "query": "MSFT Q2 25 revenue",
    "expected": "Should report Q4 FY2025 (Apr–Jun 2025) revenue = $70.1B",
    "available_periods": {
        "MSFT": [
            {"call_date": "2025-07-22", "fiscal_quarter": "Q4 FY2025", "period_end": "2025-06-30"},
            {"call_date": "2025-04-29", "fiscal_quarter": "Q3 FY2025", "period_end": "2025-03-31"},
            {"call_date": "2025-01-29", "fiscal_quarter": "Q2 FY2025", "period_end": "2024-12-31"},
        ],
    },
    "scope": SimpleRAGScope(
        tickers=["MSFT"],
        ticker_date_pairs=[("MSFT", "2025-07-22")],
        temporal_intent=TemporalIntent(type="specific_quarter", quarter=2, year=2025),
    ),
    "chunks": [
        {
            "chunk_id": "s2-fin",
            "content": (
                "FINANCIAL SUMMARY — Microsoft Q4 FY2025 Earnings Call (2025-07-22)\n\n"
                "REPORTED RESULTS (Q4 FY2025):\n"
                "- Revenue: $70.1B (up 15% YoY)\n"
                "- Intelligent Cloud Revenue: $30.3B (up 19% YoY)\n"
                "- Operating Income: $30.6B (43.6% margin)\n"
            ),
            "similarity": 0.91,
            "metadata": {
                "company_ticker": "MSFT", "company_name": "Microsoft",
                "call_date": "2025-07-22", "fiscal_quarter": "Q4 FY2025",
                "period_end": "2025-06-30", "chunk_type": "financials",
            },
        },
        {
            "chunk_id": "s2-narr",
            "content": (
                "Satya Nadella — Chairman and CEO:\n\n"
                "We had a strong Q4, with revenue of $70.1 billion growing 15% year over year. "
                "Azure and other cloud services revenue grew 33%. Our AI business is now on a "
                "$16 billion annual run rate."
            ),
            "similarity": 0.84,
            "metadata": {
                "company_ticker": "MSFT", "company_name": "Microsoft",
                "call_date": "2025-07-22", "fiscal_quarter": "Q4 FY2025",
                "period_end": "2025-06-30",
            },
        },
    ],
}

# =============================================================================
# Scenario 3: NOW vs CRM "Q4 25" — cross-company, different fiscal calendars
# NOW (FY ends Dec) Q4 FY2025 = Oct–Dec 2025, period_end Dec 31
# CRM (FY ends Jan) Q4 FY2026 = Nov–Jan 2026, period_end Jan 31
# Both are closest to CY Q4 2025 target (Dec 31) but cover different months.
# =============================================================================
SCENARIO_3 = {
    "name": "NOW vs CRM Q4 25 — cross-company fiscal calendar mismatch",
    "query": "Compare ServiceNow and Salesforce Q4 25 revenue",
    "expected": "Should compare NOW Q4 FY2025 (Oct–Dec 2025) vs CRM Q4 FY2026 (Nov–Jan 2026), noting the different periods",
    "available_periods": {
        "NOW": [
            {"call_date": "2026-01-29", "fiscal_quarter": "Q4 FY2025", "period_end": "2025-12-31"},
            {"call_date": "2025-10-29", "fiscal_quarter": "Q3 FY2025", "period_end": "2025-09-30"},
        ],
        "CRM": [
            {"call_date": "2026-03-05", "fiscal_quarter": "Q4 FY2026", "period_end": "2026-01-31"},
            {"call_date": "2025-12-03", "fiscal_quarter": "Q3 FY2026", "period_end": "2025-10-31"},
        ],
    },
    "scope": SimpleRAGScope(
        tickers=["NOW", "CRM"],
        ticker_date_pairs=[("NOW", "2026-01-29"), ("CRM", "2026-03-05")],
        temporal_intent=TemporalIntent(type="specific_quarter", quarter=4, year=2025),
    ),
    "chunks": [
        {
            "chunk_id": "s3-now-fin",
            "content": (
                "FINANCIAL SUMMARY — ServiceNow Q4 FY2025 Earnings Call (2026-01-29)\n\n"
                "REPORTED RESULTS (Q4 FY2025):\n"
                "- Subscription Revenue: $3.09B (up 21% YoY)\n"
                "- Total Revenue: $3.36B (up 21% YoY)\n"
                "- Non-GAAP Operating Margin: 31.0%\n"
            ),
            "similarity": 0.90,
            "metadata": {
                "company_ticker": "NOW", "company_name": "ServiceNow",
                "call_date": "2026-01-29", "fiscal_quarter": "Q4 FY2025",
                "period_end": "2025-12-31", "chunk_type": "financials",
            },
        },
        {
            "chunk_id": "s3-crm-fin",
            "content": (
                "FINANCIAL SUMMARY — Salesforce Q4 FY2026 Earnings Call (2026-03-05)\n\n"
                "REPORTED RESULTS (Q4 FY2026):\n"
                "- Revenue: $10.03B (up 8% YoY)\n"
                "- Subscription & Support Revenue: $9.45B (up 9% YoY)\n"
                "- Non-GAAP Operating Margin: 33.7%\n"
            ),
            "similarity": 0.89,
            "metadata": {
                "company_ticker": "CRM", "company_name": "Salesforce",
                "call_date": "2026-03-05", "fiscal_quarter": "Q4 FY2026",
                "period_end": "2026-01-31", "chunk_type": "financials",
            },
        },
    ],
}

# =============================================================================
# Scenario 4: IOT "latest revenue" — latest quarter, no specific time
# IOT FY ends January. Latest = Q4 FY2026 (Nov 2025–Jan 2026)
# Transcript says "Q4 revenue" — user didn't specify a quarter at all.
# =============================================================================
SCENARIO_4 = {
    "name": "IOT latest revenue — no time specified, latest quarter",
    "query": "Samsara revenue",
    "expected": "Should report latest quarter's revenue without confusion",
    "available_periods": {
        "IOT": [
            {"call_date": "2026-03-06", "fiscal_quarter": "Q4 FY2026", "period_end": "2026-01-31"},
            {"call_date": "2025-12-05", "fiscal_quarter": "Q3 FY2026", "period_end": "2025-10-31"},
        ],
    },
    "scope": SimpleRAGScope(
        tickers=["IOT"],
        ticker_date_pairs=[("IOT", "2026-03-06")],
        temporal_intent=TemporalIntent(type="latest"),
    ),
    "chunks": [
        {
            "chunk_id": "s4-fin",
            "content": (
                "FINANCIAL SUMMARY — Samsara Q4 FY2026 Earnings Call (2026-03-06)\n\n"
                "REPORTED RESULTS (Q4 FY2026):\n"
                "- Revenue: $346M (up 32% YoY)\n"
                "- ARR: $1.41B (up 30% YoY)\n"
                "- Non-GAAP Operating Margin: 12.8%\n"
                "- Customers >$100K ARR: 2,150\n"
            ),
            "similarity": 0.93,
            "metadata": {
                "company_ticker": "IOT", "company_name": "Samsara",
                "call_date": "2026-03-06", "fiscal_quarter": "Q4 FY2026",
                "period_end": "2026-01-31", "chunk_type": "financials",
            },
        },
    ],
}

# =============================================================================
# Scenario 5: PANW "last 4 quarters" — range query, FY ends July
# Mixed fiscal labels: Q1 FY2026, Q4 FY2025, Q3 FY2025, Q2 FY2025
# User just says "last 4 quarters", transcript uses all different FQ labels.
# =============================================================================
SCENARIO_5 = {
    "name": "PANW last 4 quarters — range query, FY ends July",
    "query": "Palo Alto Networks revenue last 4 quarters",
    "expected": "Should list 4 quarters of revenue in reverse chronological order with fiscal labels + calendar periods",
    "available_periods": {
        "PANW": [
            {"call_date": "2025-11-20", "fiscal_quarter": "Q1 FY2026", "period_end": "2025-10-31"},
            {"call_date": "2025-08-19", "fiscal_quarter": "Q4 FY2025", "period_end": "2025-07-31"},
            {"call_date": "2025-05-20", "fiscal_quarter": "Q3 FY2025", "period_end": "2025-04-30"},
            {"call_date": "2025-02-20", "fiscal_quarter": "Q2 FY2025", "period_end": "2025-01-31"},
        ],
    },
    "scope": SimpleRAGScope(
        tickers=["PANW"],
        ticker_date_pairs=[
            ("PANW", "2025-11-20"), ("PANW", "2025-08-19"),
            ("PANW", "2025-05-20"), ("PANW", "2025-02-20"),
        ],
        temporal_intent=TemporalIntent(type="range", num_quarters=4),
    ),
    "chunks": [
        {
            "chunk_id": "s5-q1fy26",
            "content": (
                "FINANCIAL SUMMARY — Palo Alto Networks Q1 FY2026 Earnings Call (2025-11-20)\n\n"
                "REPORTED RESULTS (Q1 FY2026):\n"
                "- Revenue: $2.14B (up 14% YoY)\n"
                "- NGS ARR: $4.52B (up 40% YoY)\n"
            ),
            "similarity": 0.91,
            "metadata": {
                "company_ticker": "PANW", "company_name": "Palo Alto Networks",
                "call_date": "2025-11-20", "fiscal_quarter": "Q1 FY2026",
                "period_end": "2025-10-31", "chunk_type": "financials",
            },
        },
        {
            "chunk_id": "s5-q4fy25",
            "content": (
                "FINANCIAL SUMMARY — Palo Alto Networks Q4 FY2025 Earnings Call (2025-08-19)\n\n"
                "REPORTED RESULTS (Q4 FY2025):\n"
                "- Revenue: $2.25B (up 15% YoY)\n"
                "- NGS ARR: $4.24B (up 42% YoY)\n"
            ),
            "similarity": 0.89,
            "metadata": {
                "company_ticker": "PANW", "company_name": "Palo Alto Networks",
                "call_date": "2025-08-19", "fiscal_quarter": "Q4 FY2025",
                "period_end": "2025-07-31", "chunk_type": "financials",
            },
        },
        {
            "chunk_id": "s5-q3fy25",
            "content": (
                "FINANCIAL SUMMARY — Palo Alto Networks Q3 FY2025 Earnings Call (2025-05-20)\n\n"
                "REPORTED RESULTS (Q3 FY2025):\n"
                "- Revenue: $2.08B (up 16% YoY)\n"
                "- NGS ARR: $4.00B (up 38% YoY)\n"
            ),
            "similarity": 0.87,
            "metadata": {
                "company_ticker": "PANW", "company_name": "Palo Alto Networks",
                "call_date": "2025-05-20", "fiscal_quarter": "Q3 FY2025",
                "period_end": "2025-04-30", "chunk_type": "financials",
            },
        },
        {
            "chunk_id": "s5-q2fy25",
            "content": (
                "FINANCIAL SUMMARY — Palo Alto Networks Q2 FY2025 Earnings Call (2025-02-20)\n\n"
                "REPORTED RESULTS (Q2 FY2025):\n"
                "- Revenue: $1.95B (up 14% YoY)\n"
                "- NGS ARR: $3.65B (up 37% YoY)\n"
            ),
            "similarity": 0.85,
            "metadata": {
                "company_ticker": "PANW", "company_name": "Palo Alto Networks",
                "call_date": "2025-02-20", "fiscal_quarter": "Q2 FY2025",
                "period_end": "2025-01-31", "chunk_type": "financials",
            },
        },
    ],
}


ALL_SCENARIOS = [SCENARIO_1, SCENARIO_2, SCENARIO_3, SCENARIO_4, SCENARIO_5]


async def run_scenario(scenario: dict, client, settings) -> tuple[str, str]:
    """Build prompt and call LLM for a single scenario. Returns (llm_query, answer)."""
    resolution_note = build_resolution_note(scenario["scope"], scenario["available_periods"])
    context_str = _format_context_for_prompt(scenario["chunks"])
    if resolution_note:
        context_str = resolution_note + "\n\n" + context_str

    known_tickers = format_known_tickers(ALL_COMPANIES)
    full_prompt = build_system_prompt(
        SIMPLE_RAG_SYSTEM_PROMPT,
        context=context_str,
        known_tickers=known_tickers,
        today_date=date.today().isoformat(),
    )

    llm_query = rewrite_query_for_answer_llm(
        scenario["query"], scenario["scope"], scenario["available_periods"],
    )

    input_messages = [
        {"role": "system", "content": [{"type": "input_text", "text": full_prompt}]},
        {"role": "user", "content": [{"type": "input_text", "text": llm_query}]},
    ]

    response = await client.responses.create(
        model=settings.openai_model,
        input=input_messages,
    )
    return llm_query, (response.output_text or "").strip()


async def main():
    client = get_openai_client()
    settings = get_settings()

    for i, scenario in enumerate(ALL_SCENARIOS, 1):
        print("=" * 80)
        print(f"SCENARIO {i}: {scenario['name']}")
        print(f"QUERY: {scenario['query']}")
        print(f"EXPECTED: {scenario['expected']}")
        print("-" * 80)

        # Show resolution note
        note = build_resolution_note(scenario["scope"], scenario["available_periods"])
        print(f"RESOLUTION NOTE:\n{note}")
        print("-" * 80)

        llm_query, answer = await run_scenario(scenario, client, settings)

        print(f"LLM SEES: {llm_query}")
        print("-" * 80)
        print(f"LLM RESPONSE:\n{answer}")
        print("=" * 80)
        print()


if __name__ == "__main__":
    asyncio.run(main())
