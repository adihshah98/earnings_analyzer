"""Agent API routes."""

import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.auth.dependencies import get_optional_user
from app.config import get_settings
from app.models.schemas import QueryRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent", tags=["agent"])


def _get_request_params(request: QueryRequest):
    settings = get_settings()
    return {
        "session_id": request.session_id,
        "search_mode": request.search_mode or settings.default_search_mode,
        "retrieval_threshold": (
            request.retrieval_threshold
            if request.retrieval_threshold is not None
            else settings.retrieval_threshold
        ),
    }


@router.post("/query")
async def agent_query(request: QueryRequest, user: dict | None = Depends(get_optional_user)):
    """Query the knowledge base agent. Returns SSE stream: delta events then a final 'done' event with answer, sources, confidence."""
    params = _get_request_params(request)
    user_id = user["sub"] if user else None

    async def event_stream():
        from app.agents.streaming import stream_simple_rag_or_agent

        async for event_type, payload in stream_simple_rag_or_agent(
            query=request.query,
            session_id=params["session_id"],
            search_mode=params["search_mode"],
            retrieval_threshold=params["retrieval_threshold"],
            user_id=user_id,
        ):
            if event_type == "delta":
                yield f"data: {json.dumps({'type': 'delta', 'text': payload})}\n\n"
            elif event_type == "done":
                yield f"data: {json.dumps({'type': 'done', 'payload': payload})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
