"""Tests for `app.rag.retriever` deterministic helpers (RRF merge, etc.)."""

from app.rag.retriever import _rrf_merge


class TestReciprocalRankFusion:
    def test_single_list_returns_same_order(self):
        lst = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        merged = _rrf_merge([lst])
        assert [x[0] for x in merged] == ["a", "b", "c"]

    def test_overlap_boosts_shared_items(self):
        vec = [("a", 0.9), ("b", 0.8)]
        kw = [("b", 0.7)]
        merged = _rrf_merge([vec, kw], k=60)
        assert merged[0][0] == "b"
        assert merged[1][0] == "a"

    def test_empty_input(self):
        assert _rrf_merge([]) == []
