"""Eval context: when set, retrieval and ingestion use the eval_document_chunks table."""

from contextlib import asynccontextmanager
from contextvars import ContextVar

_use_eval_chunks: ContextVar[bool] = ContextVar("_use_eval_chunks", default=False)


def use_eval_chunks() -> bool:
    """Return True if retrieval/ingestion should use the eval table."""
    return _use_eval_chunks.get()


@asynccontextmanager
async def use_eval_chunks_context():
    """Context manager to run code using the eval_document_chunks table.

    Use in eval API handlers so retrieval hits eval chunks instead of prod.
    """
    token = _use_eval_chunks.set(True)
    try:
        yield
    finally:
        _use_eval_chunks.reset(token)
