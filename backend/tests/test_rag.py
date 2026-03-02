"""Tests for the RAG pipeline."""

import pytest

from app.rag.ingestion import chunk_text, generate_doc_id
from app.rag.retriever import _rrf_merge


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
