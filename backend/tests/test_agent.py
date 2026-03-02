"""Tests for the KB agent."""

import pytest

from app.models.schemas import AgentResponse, SourceDocument
from app.tools.agent_tools import AgentDeps


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
            tool_calls_made=["search_knowledge_base"],
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


class TestAgentDeps:
    """Test agent dependency injection."""

    def test_default_deps(self):
        deps = AgentDeps()
        assert deps.session_id is None
        assert deps.user_name == "anonymous"
        assert deps.user_role == "viewer"

    def test_custom_deps(self):
        deps = AgentDeps(
            session_id="test-session-123",
            user_name="Alice",
            user_role="admin",
            user_department="engineering",
        )
        assert deps.session_id == "test-session-123"
        assert deps.user_name == "Alice"

    def test_session_propagation(self):
        """Test that session_id is properly stored."""
        deps = AgentDeps(session_id="my-session")
        assert deps.session_id == "my-session"
