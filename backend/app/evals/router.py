"""Eval API routes."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select

from app.dependencies import require_admin_key
from app.evals.datasets import list_datasets
from app.evals.retrieval import run_retrieval_eval, retrieval_eval_to_dict
from app.evals.runner import run_eval
from app.evals.context import use_eval_chunks_context
from app.models.database import get_session
from app.models.db_models import EvalResult
from app.models.schemas import EvalRunResult

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/evals", tags=["evals"], dependencies=[Depends(require_admin_key)])


@router.post("/run", response_model=EvalRunResult)
async def run_evaluation(dataset_name: str):
    """Run an eval suite against the agent.

    Uses eval_document_chunks table for RAG retrieval (same DB, isolated table).
    """
    try:
        async with use_eval_chunks_context():
            result = await run_eval(dataset_name=dataset_name)
        return result
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Eval dataset '{dataset_name}' not found",
        )
    except Exception as e:
        logger.error(f"Eval run failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrieval")
async def run_retrieval_evaluation(
    dataset_name: str,
    top_k: int = 5,
):
    """Run retrieval evals comparing vector, keyword, and hybrid search modes.

    Uses expected_sources from each eval case as relevance ground truth.
    Returns precision@k, recall@k, MRR, and hit@k per mode.
    Uses eval_document_chunks table for retrieval (same DB, isolated table).
    """
    try:
        async with use_eval_chunks_context():
            result = await run_retrieval_eval(
                dataset_name=dataset_name,
                top_k=top_k,
            )
        return retrieval_eval_to_dict(result)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Eval dataset '{dataset_name}' not found",
        )
    except Exception as e:
        logger.error(f"Retrieval eval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets", response_model=list[str])
async def list_eval_datasets():
    """List available eval datasets."""
    return list_datasets()


@router.get("/results")
async def get_eval_results(
    run_id: str | None = None,
    dataset_name: str | None = None,
    limit: int = 20,
):
    """Retrieve eval results from the database."""
    async with get_session() as session:
        stmt = select(EvalResult).order_by(EvalResult.created_at.desc()).limit(limit)
        if run_id:
            stmt = stmt.where(EvalResult.run_id == run_id)
        if dataset_name:
            stmt = stmt.where(EvalResult.dataset_name == dataset_name)

        result = await session.execute(stmt)
        rows = result.scalars().all()

    return [
        {
            "id": str(r.id),
            "run_id": r.run_id,
            "dataset_name": r.dataset_name,
            "scores": r.scores,
            "details": r.details,
            "created_at": r.created_at,
        }
        for r in rows
    ]
