"""Tests for `app.rag.ingestion` (chunking, doc IDs, transcript parsing)."""

from datetime import date

from app.rag.ingestion import (
    _is_standalone_speaker_header,
    _parse_speaker_turn,
    _split_line_on_mid_speaker,
    chunk_text,
    chunk_transcript,
    generate_doc_id,
)


class TestChunkText:
    def test_short_text_single_chunk(self):
        text = "This is a short sentence."
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_overlap_preserves_content(self):
        words = [f"word{i}" for i in range(100)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)
        all_chunk_text = " ".join(chunks)
        for word in words:
            assert word in all_chunk_text, f"Lost content: {word}"

    def test_deterministic_chunking(self):
        text = "Hello world. " * 50
        chunks1 = chunk_text(text, chunk_size=30, chunk_overlap=5)
        chunks2 = chunk_text(text, chunk_size=30, chunk_overlap=5)
        assert chunks1 == chunks2


class TestGenerateDocId:
    def test_deterministic_with_call_date(self):
        id1 = generate_doc_id("AAPL", date(2024, 7, 25))
        id2 = generate_doc_id("AAPL", date(2024, 7, 25))
        assert id1 == id2

    def test_different_tickers(self):
        id1 = generate_doc_id("AAPL", date(2024, 7, 25))
        id2 = generate_doc_id("MSFT", date(2024, 7, 25))
        assert id1 != id2

    def test_different_dates(self):
        id1 = generate_doc_id("AAPL", date(2024, 7, 25))
        id2 = generate_doc_id("AAPL", date(2024, 10, 31))
        assert id1 != id2

    def test_fallback_without_call_date(self):
        id1 = generate_doc_id("unknown", None, "My Doc", "wiki")
        id2 = generate_doc_id("unknown", None, "My Doc", "wiki")
        assert id1 == id2


class TestSpeakerLineParsers:
    def test_standalone_header(self):
        line = "Jane Doe Acme Corp – VP IR"
        assert _is_standalone_speaker_header(line) == line

    def test_standalone_header_rejects_colon(self):
        assert _is_standalone_speaker_header("Name: content") is None

    def test_parse_speaker_with_double_dash_role(self):
        sp, role, content = _parse_speaker_turn("Alice -- CFO: numbers are good")
        assert sp == "Alice" and role == "CFO" and content == "numbers are good"

    def test_parse_speaker_simple(self):
        sp, role, content = _parse_speaker_turn("Bob: hello world")
        assert sp == "Bob" and role is None and content == "hello world"

    def test_split_mid_line_speaker(self):
        line = (
            "We grew fast. William R. McDermott ServiceNow, Inc. – Chairman And next topic."
        )
        got = _split_line_on_mid_speaker(line)
        assert got is not None
        before, header, after = got
        assert "We grew fast" in before
        assert "ServiceNow" in header
        assert "next topic" in after


class TestChunkTranscript:
    def test_colon_style_single_block(self):
        text = "CFO Jane Doe: Revenue was up.\n\nCFO Jane Doe: Margins improved."
        chunks = chunk_transcript(text, chunk_size=500, chunk_overlap=64)
        assert len(chunks) >= 1
        assert all("speaker" in c for c in chunks)
        assert chunks[0]["speaker"] == "CFO Jane Doe"

    def test_unstructured_text_single_block(self):
        text = "Plain paragraph with no speaker labels at all."
        chunks = chunk_transcript(text, chunk_size=200, chunk_overlap=32)
        assert len(chunks) == 1
        assert chunks[0]["speaker"] is None
        assert "Plain paragraph" in chunks[0]["content"]
