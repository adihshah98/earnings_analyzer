"""Conversation history storage and retrieval for multi-turn agent support."""

import json
import logging
from typing import Sequence

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelRequest
from sqlalchemy import select

from app.models.database import get_session
from app.models.db_models import ConversationMessage

logger = logging.getLogger(__name__)


def _serialize_message(msg: ModelMessage) -> list:
    """Serialize a single ModelMessage to list for JSONB storage."""
    raw = ModelMessagesTypeAdapter.dump_json([msg]).decode()
    return json.loads(raw)


def _deserialize_message(data: dict | list) -> list[ModelMessage]:
    """Deserialize stored content back to ModelMessages."""
    if isinstance(data, list):
        payload = data
    else:
        payload = [data]
    return ModelMessagesTypeAdapter.validate_json(json.dumps(payload))


async def get_conversation_history(session_id: str) -> list[ModelMessage]:
    """Load conversation history for a session as PydanticAI ModelMessages.

    Args:
        session_id: The conversation session ID.

    Returns:
        List of ModelMessage (ModelRequest and ModelResponse) in chronological order.
    """
    if not session_id:
        return []

    async with get_session() as session:
        result = await session.execute(
            select(ConversationMessage.content)
            .where(ConversationMessage.session_id == session_id)
            .order_by(ConversationMessage.created_at)
        )
        rows = result.scalars().all()

    messages: list[ModelMessage] = []
    for row in rows:
        try:
            chunk = _deserialize_message(row)
            messages.extend(chunk)
        except Exception as e:
            logger.warning(f"Failed to deserialize conversation message: {e}")
            continue
    return messages


async def append_conversation_messages(
    session_id: str,
    messages: Sequence[ModelMessage],
) -> None:
    """Append new messages to a conversation session.

    Args:
        session_id: The conversation session ID.
        messages: New ModelMessage objects to store.
    """
    if not session_id or not messages:
        return

    async with get_session() as session:
        for msg in messages:
            content = _serialize_message(msg)
            role = "request" if isinstance(msg, ModelRequest) else "response"
            row = ConversationMessage(
                session_id=session_id,
                role=role,
                content=content,
            )
            session.add(row)

    logger.debug(f"Appended {len(messages)} messages to session {session_id}")


async def get_conversation_history_for_api(
    session_id: str,
) -> list[tuple[str, str, object]]:
    """Load conversation history as (role, content_text, created_at) for API response.

    Returns:
        List of (role, content_text, created_at) tuples.
    """
    if not session_id:
        return []

    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

    def extract_text(msg: ModelMessage) -> str:
        if isinstance(msg, ModelRequest):
            parts = [p.content for p in msg.parts if isinstance(p, UserPromptPart) and isinstance(p.content, str)]
            return " ".join(parts) if parts else "[user message]"
        if isinstance(msg, ModelResponse):
            parts = [p.content for p in msg.parts if isinstance(p, TextPart)]
            return " ".join(parts) if parts else "[assistant message]"
        return str(msg)

    async with get_session() as session:
        result = await session.execute(
            select(ConversationMessage.role, ConversationMessage.content, ConversationMessage.created_at)
            .where(ConversationMessage.session_id == session_id)
            .order_by(ConversationMessage.created_at)
        )
        rows = result.mappings().all()

    entries: list[tuple[str, str, object]] = []
    for row in rows:
        role = "user" if row["role"] == "request" else "assistant"
        msgs = _deserialize_message(row["content"])
        content = extract_text(msgs[0]) if msgs else ""
        entries.append((role, content, row["created_at"]))
    return entries


def get_recent_user_queries(messages: list[ModelMessage], limit: int = 5) -> list[str]:
    """Extract recent user queries from conversation history for RAG context.

    Args:
        messages: Conversation history as ModelMessages.
        limit: Maximum number of recent queries to return.

    Returns:
        List of user query strings (most recent last).
    """
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    queries: list[str] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    queries.append(part.content.strip())
                    if len(queries) > limit:
                        queries.pop(0)
    return queries[-limit:] if len(queries) > limit else queries
