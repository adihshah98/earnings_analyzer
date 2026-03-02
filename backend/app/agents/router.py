"""Agent API routes."""

import logging

from fastapi import APIRouter

from app.agents.kb_agent import query_agent
from app.config import get_settings
from app.models.schemas import AgentResponse, QueryRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/query", response_model=AgentResponse)
async def agent_query(request: QueryRequest):
    """Query the knowledge base agent.

    Only ``query`` is required in the request body.  All other fields
    fall back to application-level defaults defined in ``Settings``.
    """
    settings = get_settings()

    session_id = request.session_id
    company_ticker = request.company_ticker or settings.default_company_ticker
    search_mode = request.search_mode or settings.default_search_mode
    retrieval_threshold = (
        request.retrieval_threshold
        if request.retrieval_threshold is not None
        else settings.retrieval_threshold
    )

    filter_metadata = {"company_ticker": company_ticker} if company_ticker else None

    response = await query_agent(
        query=request.query,
        session_id=session_id,
        search_mode=search_mode,
        retrieval_threshold=retrieval_threshold,
        filter_metadata=filter_metadata,
    )
    return response
