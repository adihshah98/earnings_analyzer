"""Tests for `app.agents.prompt_utils` (sanitization, system prompt building)."""

from app.agents.prompt_utils import (
    _sanitize_for_prompt,
    build_system_prompt,
    format_known_tickers,
)


class TestSanitizeForPrompt:
    def test_collapses_whitespace(self):
        assert _sanitize_for_prompt("  a\n\tb  ") == "a b"

    def test_truncates(self):
        long_s = "x" * 100
        assert len(_sanitize_for_prompt(long_s, max_len=10)) == 10


class TestFormatKnownTickers:
    def test_empty(self):
        assert format_known_tickers([]) == "none"

    def test_formats_pairs(self):
        out = format_known_tickers([{"ticker": "NOW", "name": "ServiceNow"}])
        assert "NOW" in out and "ServiceNow" in out


class TestBuildSystemPrompt:
    def test_format_branch(self):
        template = "ctx={context}|k={known_tickers}|d={today_date}"
        out = build_system_prompt(
            template,
            context="X",
            known_tickers="T",
            today_date="D",
        )
        assert out == "ctx=X|k=T|d=D"

    def test_keyerror_fallback(self):
        template = "{context} and {known_tickers} on {today_date}"
        out = build_system_prompt(
            template,
            context="C",
            known_tickers="K",
            today_date="2025-01-01",
        )
        assert "C" in out and "K" in out
