"""Tests for AgentResponse schema (used by simple RAG)."""

import pytest

from app.models.schemas import AgentResponse, SourceDocument


class TestAgentResponse:
    """Test structured output validation."""

    def test_valid_response(self):
        """Test that a well-formed response passes validation."""
        response = AgentResponse(
            answer="The PTO policy allows 20 days per year.",
            confidence=0.95,
            sources=[
                SourceDocument(
                    chunk_id="abc123",
                    content="Employees get 20 days PTO...",
                    similarity=0.92,
                )
            ],
            reasoning="Found direct answer in HR handbook.",
            tool_calls_made=[],
        )
        assert response.answer
        assert response.confidence == 0.95
        assert len(response.sources) == 1

    def test_confidence_out_of_range(self):
        """Test that confidence scores outside 0-1 are rejected."""
        with pytest.raises(Exception):
            AgentResponse(
                answer="Test answer",
                confidence=5.0,  # Invalid! Should be 0.0-1.0
                sources=[],
            )

    def test_negative_confidence(self):
        """Negative confidence should be rejected."""
        with pytest.raises(Exception):
            AgentResponse(
                answer="Test answer",
                confidence=-0.5,
                sources=[],
            )


