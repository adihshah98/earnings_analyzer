"""Tests for the RAG pipeline."""

from datetime import date

import pytest

from app.agents.simple_rag import (
    TemporalIntent,
    _fix_last_year,
    _resolve_temporal,
)
from app.rag.ingestion import chunk_text, generate_doc_id
from app.rag.retriever import _rrf_merge


class TestLastYearTemporal:
    """'Last year' maps to rolling 4 calendar quarters; see _fix_last_year."""

    def test_fix_last_year_sets_rolling_four(self):
        t = TemporalIntent(type="latest")
        out = _fix_last_year("What did NOW say about Veza last year?", None, t)
        assert out.type == "range"
        assert out.num_quarters == 4
        assert out.start_year is None and out.end_year is None

    def test_fix_last_year_skips_bare_year_in_query(self):
        t = TemporalIntent(type="latest")
        out = _fix_last_year("NOW revenue in 2023", None, t)
        assert out.type == "latest"

    def test_resolve_temporal_last_four_cy_quarters_march_2025(self):
        """As of Mar 2025, last completed CY quarter is Q4 2024 → four quarters are all of CY 2024."""
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


class TestChunking:
    """Test document chunking logic."""

    def test_short_text_single_chunk(self):
        """Short text should remain as a single chunk."""
        text = "This is a short sentence."
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_overlap_preserves_content(self):
        """Chunks should overlap so no content is lost."""
        # Create a text with known tokens
        words = [f"word{i}" for i in range(100)]
        text = " ".join(words)

        chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)

        # Reconstruct: all original content should be recoverable
        all_chunk_text = " ".join(chunks)
        for word in words:
            assert word in all_chunk_text, f"Lost content: {word}"

    def test_deterministic_chunking(self):
        """Same input should produce same chunks."""
        text = "Hello world. " * 50
        chunks1 = chunk_text(text, chunk_size=30, chunk_overlap=5)
        chunks2 = chunk_text(text, chunk_size=30, chunk_overlap=5)
        assert chunks1 == chunks2


class TestDocId:
    """Test document ID generation."""

    def test_deterministic_with_call_date(self):
        from datetime import date

        id1 = generate_doc_id("AAPL", date(2024, 7, 25))
        id2 = generate_doc_id("AAPL", date(2024, 7, 25))
        assert id1 == id2

    def test_different_tickers(self):
        from datetime import date

        id1 = generate_doc_id("AAPL", date(2024, 7, 25))
        id2 = generate_doc_id("MSFT", date(2024, 7, 25))
        assert id1 != id2

    def test_different_dates(self):
        from datetime import date

        id1 = generate_doc_id("AAPL", date(2024, 7, 25))
        id2 = generate_doc_id("AAPL", date(2024, 10, 31))
        assert id1 != id2

    def test_fallback_without_call_date(self):
        id1 = generate_doc_id("unknown", None, "My Doc", "wiki")
        id2 = generate_doc_id("unknown", None, "My Doc", "wiki")
        assert id1 == id2


class TestRRF:
    """Test Reciprocal Rank Fusion merge."""

    def test_rrf_merge_single_list(self):
        """Single list returns same order."""
        lst = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        merged = _rrf_merge([lst])
        assert [x[0] for x in merged] == ["a", "b", "c"]

    def test_rrf_merge_boosts_overlap(self):
        """Items appearing in both lists get higher RRF than those in one."""
        vec = [("a", 0.9), ("b", 0.8)]  # a rank 0, b rank 1
        kw = [("b", 0.7)]  # b only, rank 0
        merged = _rrf_merge([vec, kw], k=60)
        # a: 1/61 only. b: 1/62 + 1/61 -> b has higher RRF
        assert merged[0][0] == "b"
        assert merged[1][0] == "a"

    def test_rrf_merge_empty(self):
        """Empty list returns empty."""
        merged = _rrf_merge([])
        assert merged == []
