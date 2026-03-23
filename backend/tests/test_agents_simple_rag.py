"""Tests for deterministic behavior in `app.agents.simple_rag` (temporal, prompts, chunk helpers)."""

from datetime import date
from types import SimpleNamespace

from app.agents.simple_rag import (
    SimpleRAGScope,
    TemporalIntent,
    _extract_text_from_responses_output,
    _find_closest_period,
    _fix_bare_year,
    _fix_last_year,
    _resolve_temporal,
    _scope_cache_key,
    _strip_json_fences,
    build_resolution_note,
    reorder_chunks_for_range,
    rewrite_query_for_answer_llm,
    trim_chunks_to_token_budget,
    _format_context_for_prompt,
)


class TestStripJsonFences:
    def test_plain_json(self):
        assert _strip_json_fences('{"a": 1}') == '{"a": 1}'

    def test_fenced_json(self):
        raw = '```json\n{"tickers": ["NOW"]}\n```'
        assert _strip_json_fences(raw) == '{"tickers": ["NOW"]}'

    def test_fenced_without_json_label(self):
        raw = '```\n{"x": true}\n```'
        assert _strip_json_fences(raw) == '{"x": true}'


class TestFixBareYear:
    def test_bare_year_forces_full_year_range(self):
        t = TemporalIntent(type="latest")
        out = _fix_bare_year("Meta 2024 revenue", t)
        assert out.type == "range"
        assert out.start_year == 2024 and out.end_year == 2024
        assert out.num_quarters is None

    def test_skips_when_quarter_year_present(self):
        t = TemporalIntent(type="latest")
        out = _fix_bare_year("revenue Q3 2024", t)
        assert out.type == "latest"

    def test_preserves_existing_anchored_range(self):
        t = TemporalIntent(type="range", start_year=2023, end_year=2023)
        out = _fix_bare_year("something 2024 in text", t)
        assert out.start_year == 2023


class TestFixLastYear:
    """'Last year' maps to rolling 4 calendar quarters; see _fix_last_year."""

    def test_sets_rolling_four(self):
        t = TemporalIntent(type="latest")
        out = _fix_last_year("What did NOW say about Veza last year?", None, t)
        assert out.type == "range"
        assert out.num_quarters == 4
        assert out.start_year is None and out.end_year is None

    def test_skips_bare_year_in_query(self):
        t = TemporalIntent(type="latest")
        out = _fix_last_year("NOW revenue in 2023", None, t)
        assert out.type == "latest"

    def test_resolve_temporal_last_four_cy_quarters_march_2025(self):
        """Last completed CY quarter is Q4 2024 → four quarters are all of CY 2024."""
        temporal = TemporalIntent(type="range", num_quarters=4)
        periods = {
            "NOW": [
                {"call_date": "2025-01-29", "period_end": "2024-12-31"},
                {"call_date": "2024-10-29", "period_end": "2024-09-30"},
                {"call_date": "2024-07-24", "period_end": "2024-06-30"},
                {"call_date": "2024-04-24", "period_end": "2024-03-31"},
            ],
        }
        pairs = _resolve_temporal(["NOW"], temporal, periods, today=date(2025, 3, 22))
        assert pairs is not None
        assert {p[1] for p in pairs} == {
            "2025-01-29",
            "2024-10-29",
            "2024-07-24",
            "2024-04-24",
        }


class TestFindClosestPeriod:
    def test_single_candidate(self):
        periods = [{"call_date": "2025-01-01", "period_end": "2024-12-31"}]
        got = _find_closest_period(periods, date(2024, 12, 31))
        assert got == periods[0]

    def test_picks_min_abs_delta(self):
        periods = [
            {"call_date": "a", "period_end": "2024-06-30"},
            {"call_date": "b", "period_end": "2024-12-31"},
        ]
        got = _find_closest_period(periods, date(2024, 11, 15))
        assert got["call_date"] == "b"


class TestResolveTemporal:
    def test_unspecified_returns_none(self):
        assert (
            _resolve_temporal(
                ["NOW"],
                TemporalIntent(type="unspecified"),
                {"NOW": [{"call_date": "2025-01-01", "period_end": "2024-12-31"}]},
                today=date(2025, 3, 22),
            )
            is None
        )

    def test_specific_quarter_missing_quarter_returns_none(self):
        assert (
            _resolve_temporal(
                ["NOW"],
                TemporalIntent(type="specific_quarter", quarter=None, year=2025),
                {"NOW": [{"call_date": "2025-01-01", "period_end": "2024-12-31"}]},
            )
            is None
        )

    def test_latest_uses_first_period_per_ticker(self):
        periods = {
            "NOW": [
                {"call_date": "2025-01-29", "period_end": "2024-12-31"},
                {"call_date": "2024-10-01", "period_end": "2024-09-30"},
            ],
        }
        pairs = _resolve_temporal(["NOW"], TemporalIntent(type="latest"), periods)
        assert pairs == [("NOW", "2025-01-29")]

    def test_anchored_range_spans_years(self):
        periods = {
            "X": [
                {"call_date": "d1", "period_end": "2023-03-31"},
                {"call_date": "d2", "period_end": "2024-06-30"},
            ],
        }
        temporal = TemporalIntent(
            type="range",
            start_year=2023,
            end_year=2024,
            start_quarter=1,
            end_quarter=2,
        )
        pairs = _resolve_temporal(["X"], temporal, periods, today=date(2025, 1, 1))
        call_dates = {p[1] for p in pairs}
        assert "d1" in call_dates and "d2" in call_dates


class TestScopeCacheKey:
    def test_normalization_and_truncation(self):
        companies = [{"ticker": "A", "name": "Alpha"}]
        k1 = _scope_cache_key("  HELLO  ", companies, None)
        k2 = _scope_cache_key("hello", companies, None)
        assert k1[0] == k2[0]

    def test_context_tail(self):
        companies = [{"ticker": "A", "name": "Alpha"}]
        ctx = ["a", "b", "c", "d", "e", "f"]
        k = _scope_cache_key("q", companies, ctx)
        assert k[0] == "q"
        assert k[2] == ("b", "c", "d", "e", "f")


class TestExtractTextFromResponsesOutput:
    def test_empty(self):
        assert _extract_text_from_responses_output([]) == ""

    def test_message_with_output_text(self):
        out = [
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="part1")],
            ),
        ]
        assert _extract_text_from_responses_output(out) == "part1"


class TestTrimChunksToTokenBudget:
    def test_no_trim_when_under_budget(self):
        chunks = [
            {"content": "short", "metadata": {}},
            {"content": "tiny", "metadata": {"chunk_type": "financials"}},
        ]
        out = trim_chunks_to_token_budget(chunks, budget=10_000)
        assert out is chunks

    def test_trims_when_over_budget(self):
        long_text = "word " * 2000
        chunks = [
            {"content": long_text, "metadata": {}},
            {"content": long_text, "metadata": {}},
        ]
        out = trim_chunks_to_token_budget(chunks, budget=50)
        assert len(out) < len(chunks)


class TestReorderChunksForRange:
    def test_non_range_unchanged(self):
        chunks = [{"metadata": {"call_date": "2025-01-01"}}]
        scope = SimpleRAGScope(temporal_intent=TemporalIntent(type="latest"))
        assert reorder_chunks_for_range(chunks, scope) is chunks

    def test_range_sorts_by_period(self):
        scope = SimpleRAGScope(temporal_intent=TemporalIntent(type="range", num_quarters=2))
        chunks = [
            {"metadata": {"period_end": "2024-06-30", "chunk_type": "narrative"}},
            {"metadata": {"period_end": "2025-03-31", "chunk_type": "financials"}},
        ]
        out = reorder_chunks_for_range(chunks, scope)
        assert out[0]["metadata"]["period_end"] == "2025-03-31"


class TestBuildResolutionNote:
    def test_empty_without_pairs(self):
        scope = SimpleRAGScope(
            tickers=["NOW"],
            temporal_intent=TemporalIntent(type="latest"),
        )
        assert build_resolution_note(scope, {}) == ""

    def test_includes_mapping_for_specific_quarter(self):
        scope = SimpleRAGScope(
            tickers=["CRM"],
            ticker_date_pairs=[("CRM", "2026-03-05")],
            temporal_intent=TemporalIntent(type="specific_quarter", quarter=4, year=2025),
        )
        periods = {
            "CRM": [
                {
                    "call_date": "2026-03-05",
                    "fiscal_quarter": "Q4 FY2026",
                    "period_end": "2026-01-31",
                },
            ],
        }
        note = build_resolution_note(scope, periods)
        assert "TEMPORAL RESOLUTION" in note
        assert "Q4 FY2026" in note
        assert "MAPPING" in note

    def test_full_year_range_adds_aggregation_instructions(self):
        scope = SimpleRAGScope(
            ticker_date_pairs=[("X", "d1")],
            temporal_intent=TemporalIntent(
                type="range",
                start_year=2024,
                end_year=2024,
                start_quarter=1,
                end_quarter=4,
            ),
        )
        periods = {"X": [{"call_date": "d1", "fiscal_quarter": "Q4 FY2024", "period_end": "2024-12-31"}]}
        note = build_resolution_note(scope, periods, query="annual revenue 2024")
        assert "FULL-YEAR QUERY" in note
        assert "STEP 1" in note


class TestRewriteQueryForAnswerLlm:
    def test_no_pairs_returns_original(self):
        scope = SimpleRAGScope(
            tickers=["NOW"],
            temporal_intent=TemporalIntent(type="specific_quarter", quarter=1, year=2025),
        )
        q = "NOW revenue"
        assert rewrite_query_for_answer_llm(q, scope, {}) == q

    def test_appends_fiscal_label_single_ticker(self):
        scope = SimpleRAGScope(
            ticker_date_pairs=[("MSFT", "2025-07-22")],
            temporal_intent=TemporalIntent(type="specific_quarter", quarter=2, year=2025),
        )
        periods = {
            "MSFT": [
                {
                    "call_date": "2025-07-22",
                    "fiscal_quarter": "Q4 FY2025",
                    "period_end": "2025-06-30",
                },
            ],
        }
        out = rewrite_query_for_answer_llm("MSFT Q2 25 revenue", scope, periods)
        assert "Q4 FY2025" in out
        assert out.startswith("MSFT Q2 25 revenue")

    def test_range_intent_not_rewritten(self):
        scope = SimpleRAGScope(
            ticker_date_pairs=[("PANW", "2025-11-20")],
            temporal_intent=TemporalIntent(type="range", num_quarters=4),
        )
        q = "PANW revenue trend"
        assert rewrite_query_for_answer_llm(q, scope, {}) == q


class TestFormatContextForPrompt:
    def test_formats_source_block(self):
        chunks = [
            {
                "content": "Body text",
                "similarity": 0.91,
                "metadata": {
                    "company_ticker": "NOW",
                    "company_name": "ServiceNow",
                    "call_date": "2026-01-29",
                    "fiscal_quarter": "Q4 FY2025",
                    "period_end": "2025-12-31",
                },
            },
        ]
        s = _format_context_for_prompt(chunks)
        assert "[Source 1:" in s
        assert "NOW" in s
        assert "Body text" in s
        assert "0.91" in s
