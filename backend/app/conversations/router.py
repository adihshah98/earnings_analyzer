"""Conversation history API routes."""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.conversations.service import get_conversation_history_for_api

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/conversations", tags=["conversations"])


class HistoryEntry(BaseModel):
    """Single entry in conversation history."""

    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    created_at: datetime | None = Field(None, description="When the message was created")


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
