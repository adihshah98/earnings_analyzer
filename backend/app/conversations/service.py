"""Conversation history storage and retrieval for multi-turn agent support."""

import json
import logging
from typing import Sequence

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelRequest
from sqlalchemy import delete, func, nulls_last, select

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
            .order_by(
                nulls_last(ConversationMessage.position.asc()),
                ConversationMessage.created_at.asc(),
                ConversationMessage.id.asc(),
            )
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
    user_id: str | None = None,
) -> None:
    """Append new messages to a conversation session.

    Args:
        session_id: The conversation session ID.
        messages: New ModelMessage objects to store.
    """
    if not session_id or not messages:
        return

    async with get_session() as session:
        next_pos_result = await session.execute(
            select(func.coalesce(func.max(ConversationMessage.position), 0))
            .where(ConversationMessage.session_id == session_id)
        )
        next_pos = (next_pos_result.scalar_one() or 0) + 1
        import uuid as _uuid
        parsed_user_id = _uuid.UUID(user_id) if user_id else None
        for msg in messages:
            content = _serialize_message(msg)
            role = "request" if isinstance(msg, ModelRequest) else "response"
            row = ConversationMessage(
                session_id=session_id,
                user_id=parsed_user_id,
                role=role,
                content=content,
                position=next_pos,
            )
            next_pos += 1
            session.add(row)

    logger.debug(f"Appended {len(messages)} messages to session {session_id}")


async def append_conversation_turn(
    session_id: str,
    user_text: str,
    assistant_text: str,
    user_id: str | None = None,
    sources: list | None = None,
) -> None:
    """Append a user+assistant turn to a session. Wrapper for simple text persistence.

    Args:
        session_id: The conversation session ID.
        user_text: The user's message text.
        assistant_text: The assistant's response text.
        sources: Optional list of source dicts to store with the assistant message.
    """
    if not session_id:
        return
    import uuid as _uuid
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

    request = ModelRequest(parts=[UserPromptPart(content=user_text)])
    response = ModelResponse(parts=[TextPart(content=assistant_text)])
    parsed_user_id = _uuid.UUID(user_id) if user_id else None

    async with get_session() as session:
        next_pos_result = await session.execute(
            select(func.coalesce(func.max(ConversationMessage.position), 0))
            .where(ConversationMessage.session_id == session_id)
        )
        next_pos = (next_pos_result.scalar_one() or 0) + 1

        req_row = ConversationMessage(
            session_id=session_id,
            user_id=parsed_user_id,
            role="request",
            content=_serialize_message(request),
            title=user_text[:100],
            position=next_pos,
        )
        session.add(req_row)

        resp_row = ConversationMessage(
            session_id=session_id,
            user_id=parsed_user_id,
            role="response",
            content=_serialize_message(response),
            sources=sources,
            position=next_pos + 1,
        )
        session.add(resp_row)

    logger.debug(f"Appended turn to session {session_id}")


async def list_sessions(user_id: str | None = None) -> list[tuple[str, object, str | None]]:
    """List conversation session IDs with their last activity time and title.

    If user_id is provided, returns only sessions belonging to that user.

    Returns:
        List of (session_id, updated_at, title) tuples, newest first.
        Title is read directly from the stored title column — no deserialization needed.
    """
    import uuid as _uuid

    async with get_session() as session:
        q = select(
            ConversationMessage.session_id,
            func.max(ConversationMessage.created_at).label("updated_at"),
        )
        if user_id:
            q = q.where(ConversationMessage.user_id == _uuid.UUID(user_id))
        q = q.group_by(ConversationMessage.session_id)
        subq = q.subquery()
        result = await session.execute(
            select(subq.c.session_id, subq.c.updated_at).order_by(
                subq.c.updated_at.desc().nulls_last()
            )
        )
        rows = list(result.all())
        session_ids = [r[0] for r in rows]

        if not session_ids:
            return []

        # Fetch first title per session using DISTINCT ON (no deserialization).
        # Falls back to None for sessions created before migration 015.
        first_title_q = (
            select(
                ConversationMessage.session_id,
                ConversationMessage.title,
            )
            .where(
                ConversationMessage.session_id.in_(session_ids),
                ConversationMessage.role == "request",
                ConversationMessage.title.isnot(None),
            )
            .distinct(ConversationMessage.session_id)
            .order_by(
                ConversationMessage.session_id,
                nulls_last(ConversationMessage.position.asc()),
                ConversationMessage.created_at.asc(),
            )
        )
        title_result = await session.execute(first_title_q)
        titles = {r[0]: r[1] for r in title_result.all()}

    return [(sid, updated_at, titles.get(sid)) for sid, updated_at in rows]


async def delete_session(session_id: str) -> None:
    """Delete all messages for a conversation session.

    Args:
        session_id: The conversation session ID to delete.
    """
    if not session_id:
        return
    async with get_session() as session:
        await session.execute(delete(ConversationMessage).where(ConversationMessage.session_id == session_id))
    logger.debug(f"Deleted session {session_id}")


async def get_conversation_history_for_api(
    session_id: str,
) -> list[tuple[str, str, object, list | None]]:
    """Load conversation history as (role, content_text, created_at, sources) for API response.

    Returns:
        List of (role, content_text, created_at, sources) tuples.
        sources is only non-None for assistant (response) messages.
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
            select(
                ConversationMessage.role,
                ConversationMessage.content,
                ConversationMessage.created_at,
                ConversationMessage.sources,
            )
            .where(ConversationMessage.session_id == session_id)
            .order_by(
                nulls_last(ConversationMessage.position.asc()),
                ConversationMessage.created_at.asc(),
                ConversationMessage.id.asc(),
            )
        )
        rows = result.mappings().all()

    entries: list[tuple[str, str, object, list | None]] = []
    for row in rows:
        role = "user" if row["role"] == "request" else "assistant"
        msgs = _deserialize_message(row["content"])
        content = extract_text(msgs[0]) if msgs else ""
        sources = row["sources"] if role == "assistant" else None
        entries.append((role, content, row["created_at"], sources))
    return entries


def get_recent_turns(messages: list[ModelMessage], limit: int = 5) -> list[tuple[str, str]]:
    """Extract recent (user_query, assistant_answer) pairs from conversation history.

    Returns:
        List of (query, answer) tuples, most recent last.
    """
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

    pairs: list[tuple[str, str]] = []
    msgs = list(messages)
    i = 0
    while i < len(msgs):
        msg = msgs[i]
        if isinstance(msg, ModelRequest):
            query = " ".join(
                p.content for p in msg.parts
                if isinstance(p, UserPromptPart) and isinstance(p.content, str)
            ).strip()
            answer = ""
            if i + 1 < len(msgs) and isinstance(msgs[i + 1], ModelResponse):
                answer = " ".join(
                    p.content for p in msgs[i + 1].parts if isinstance(p, TextPart)
                ).strip()
                i += 2
            else:
                i += 1
            if query:
                pairs.append((query, answer))
        else:
            i += 1
    return pairs[-limit:]


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
