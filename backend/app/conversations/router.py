"""Conversation history API routes."""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.conversations.service import (
    delete_session,
    get_conversation_history_for_api,
    list_sessions,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/conversations", tags=["conversations"])


class SessionSummary(BaseModel):
    """Summary of a conversation session for list view."""

    session_id: str = Field(..., description="Session ID")
    updated_at: datetime | None = Field(None, description="Last activity time")


class HistoryEntry(BaseModel):
    """Single entry in conversation history."""

    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    created_at: datetime | None = Field(None, description="When the message was created")


@router.get("/sessions", response_model=list[SessionSummary])
async def get_sessions():
    """List all conversation sessions, newest first."""
    rows = await list_sessions()
    return [
        SessionSummary(session_id=session_id, updated_at=updated_at)
        for session_id, updated_at in rows
    ]


@router.delete("/{session_id}")
async def delete_conversation(session_id: str):
    """Delete a conversation session and all its messages."""
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    await delete_session(session_id)
    return {"ok": True}


@router.get("/{session_id}/history", response_model=list[HistoryEntry])
async def get_history(session_id: str):
    """Get conversation history for a session.

    Returns messages in chronological order. Each entry includes
    role (user/assistant) and the message content.
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    rows = await get_conversation_history_for_api(session_id)
    return [
        HistoryEntry(role=role, content=content, created_at=created_at)
        for role, content, created_at in rows
    ]
