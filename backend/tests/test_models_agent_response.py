"""Tests for `app.models.schemas` agent-facing models (AgentResponse, SourceDocument)."""

import pytest

from app.models.schemas import AgentResponse, Citation, SourceDocument


class TestSourceDocument:
    def test_minimal_source(self):
        src = SourceDocument(
            chunk_id="c1",
            content="snippet",
            similarity=0.9,
        )
        assert src.chunk_id == "c1"
        assert src.metadata == {}
        assert src.cited_spans == []


class TestAgentResponse:
    def test_valid_response(self):
        response = AgentResponse(
            answer="The policy allows 20 days per year.",
            sources=[
                SourceDocument(
                    chunk_id="abc123",
                    content="Employees get 20 days...",
                    similarity=0.92,
                )
            ],
            reasoning="Found in handbook.",
            tool_calls_made=[],
        )
        assert response.answer
        assert len(response.sources) == 1
        assert response.citations == []

    def test_with_citations(self):
        r = AgentResponse(
            answer="See source 1.",
            sources=[
                SourceDocument(chunk_id="a", content="quote me", similarity=1.0, source_index=1)
            ],
            citations=[Citation(source_index=1, quote="quote me")],
        )
        assert len(r.citations) == 1
        assert r.citations[0].quote == "quote me"

    def test_citation_source_index_below_one_rejected(self):
        with pytest.raises(Exception):
            Citation(source_index=0, quote="x")

    def test_citation_empty_quote_rejected(self):
        with pytest.raises(Exception):
            Citation(source_index=1, quote="")
