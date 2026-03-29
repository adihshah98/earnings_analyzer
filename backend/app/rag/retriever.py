"""pgvector retrieval for RAG pipeline via SQLAlchemy + asyncpg."""

import asyncio
import logging
import re
import time
from uuid import UUID

from typing import Any, Literal

from sqlalchemy import and_, desc, func, or_, select

from app.config import get_settings
from app.evals.context import use_eval_chunks
from app.models.database import get_session
from app.models.db_models import DocumentChunk, EvalDocumentChunk
from app.rag.embeddings import generate_embedding
from app.rag.ticker_map import TICKER_NAMES


def _get_chunk_model():
    """Return DocumentChunk or EvalDocumentChunk based on eval context."""
    return EvalDocumentChunk if use_eval_chunks() else DocumentChunk

logger = logging.getLogger(__name__)

SearchMode = Literal["vector", "keyword", "hybrid"]
RRF_K = 60  # Reciprocal Rank Fusion constant



def rrf_merge(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """Merge ranked lists using Reciprocal Rank Fusion.

    Each list is [(chunk_id, score), ...] ordered by relevance (best first).
    RRF score = sum(1 / (k + rank + 1)) over all lists where the item appears.

    Args:
        ranked_lists: List of ranked result lists. Each list has (chunk_id, original_score).
        k: RRF constant (default 60).

    Returns:
        List of (chunk_id, best_orig_similarity) sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    best_orig: dict[str, float] = {}

    for lst in ranked_lists:
        for rank, (chunk_id, orig_score) in enumerate(lst):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)
            best_orig[chunk_id] = max(best_orig.get(chunk_id, 0), orig_score)

    merged = sorted(
        rrf_scores.keys(),
        key=lambda x: rrf_scores[x],
        reverse=True,
    )
    return [(cid, best_orig[cid]) for cid in merged]






def _apply_metadata_filters(stmt, filter_metadata: dict[str, Any] | None, chunk_model=None):
    """Apply metadata filters to a SQLAlchemy select statement.

    Special keys:
        ``_ticker_date_pairs``: list of (ticker, call_date) tuples — most specific filter.
        ``_tickers``: list of ticker strings — broader ticker-only filter.
    """
    Chunk = chunk_model or _get_chunk_model()
    if not filter_metadata:
        return stmt

    # _ticker_date_pairs is the most specific: filter to exact (ticker, call_date) combos
    ticker_date_pairs = filter_metadata.get("_ticker_date_pairs")
    if ticker_date_pairs:
        conditions = [
            and_(Chunk.company_ticker == t, Chunk.call_date == d)
            for t, d in ticker_date_pairs
        ]
        stmt = stmt.where(or_(*conditions))
    else:
        tickers = filter_metadata.get("_tickers")
        if tickers:
            stmt = stmt.where(Chunk.company_ticker.in_(tickers))

    for key, value in filter_metadata.items():
        if key.startswith("_"):
            continue
        if key == "company_ticker":
            stmt = stmt.where(Chunk.company_ticker == str(value))
        elif key == "call_date":
            stmt = stmt.where(Chunk.call_date == str(value))
        else:
            stmt = stmt.where(Chunk.chunk_metadata[key].astext == str(value))
    return stmt


async def _retrieve_vector(
    search_query: str,
    top_k: int,
    threshold: float,
    filter_metadata: dict[str, Any] | None,
    query_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve chunks using vector (cosine similarity) search.

    If query_embedding is provided, skips the embedding API call (for parallelization).
    """
    Chunk = _get_chunk_model()
    if query_embedding is None:
        t0 = time.perf_counter()
        query_embedding = await generate_embedding(search_query)
        logger.info("[latency] generate_embedding: %.3fs", time.perf_counter() - t0)
    cosine_dist = Chunk.embedding.cosine_distance(query_embedding)
    similarity = 1 - cosine_dist

    stmt = (
        select(
            Chunk.id,
            Chunk.content,
            Chunk.chunk_metadata,
            similarity.label("similarity"),
        )
        .where(Chunk.embedding.isnot(None))
        .where(similarity > threshold)
        .order_by(cosine_dist)
        .limit(top_k)
    )
    stmt = _apply_metadata_filters(stmt, filter_metadata, Chunk)

    t0 = time.perf_counter()
    async with get_session() as session:
        result = await session.execute(stmt)
        rows = result.mappings().all()
    logger.info("[latency] _retrieve_vector DB: %.3fs", time.perf_counter() - t0)

    return [
        {
            "id": row["id"],
            "content": row["content"],
            "metadata": row["chunk_metadata"] or {},
            "similarity": float(row["similarity"]),
        }
        for row in rows
    ]


async def _retrieve_keyword(
    query: str,
    top_k: int,
    filter_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve chunks using PostgreSQL full-text search (ts_rank).

    Uses websearch_to_tsquery (phrase-aware, AND-based) for precise matching,
    with fallback to plainto_tsquery on parse errors.
    """
    Chunk = _get_chunk_model()

    def _build_stmt(ts_query):
        rank_expr = func.coalesce(func.ts_rank(Chunk.content_tsv, ts_query), 0)
        stmt = (
            select(Chunk.id, Chunk.content, Chunk.chunk_metadata, rank_expr.label("similarity"))
            .where(Chunk.content_tsv.op("@@")(ts_query))
            .order_by(desc(rank_expr))
            .limit(top_k)
        )
        return _apply_metadata_filters(stmt, filter_metadata, Chunk), rank_expr

    t0 = time.perf_counter()
    async with get_session() as session:
        try:
            ts_query = func.websearch_to_tsquery("english", query)
            stmt, _ = _build_stmt(ts_query)
            result = await session.execute(stmt)
        except Exception as e:
            err = str(e).lower()
            if "content_tsv" in err:
                # Migration not yet run — compute tsvector on the fly
                tsv = func.to_tsvector("english", Chunk.content)
                ts_query_fb = func.websearch_to_tsquery("english", query)
                rank_expr_fb = func.coalesce(func.ts_rank(tsv, ts_query_fb), 0)
                stmt_fb = (
                    select(Chunk.id, Chunk.content, Chunk.chunk_metadata, rank_expr_fb.label("similarity"))
                    .where(tsv.op("@@")(ts_query_fb))
                    .order_by(desc(rank_expr_fb))
                    .limit(top_k)
                )
                stmt_fb = _apply_metadata_filters(stmt_fb, filter_metadata, Chunk)
                result = await session.execute(stmt_fb)
            else:
                # Any other error: fall back to plainto_tsquery (AND-based, very permissive)
                ts_query_fb = func.plainto_tsquery("english", query)
                stmt_fb, _ = _build_stmt(ts_query_fb)
                result = await session.execute(stmt_fb)

        rows = result.mappings().all()
    logger.info("[latency] _retrieve_keyword DB: %.3fs", time.perf_counter() - t0)

    # Normalize ts_rank to [0,1] for display (ts_rank can be >1)
    max_rank = max((r["similarity"] or 0) for r in rows) if rows else 1.0
    if max_rank <= 0:
        max_rank = 1.0

    return [
        {
            "id": row["id"],
            "content": row["content"],
            "metadata": row["chunk_metadata"] or {},
            "similarity": min(1.0, float(row["similarity"] or 0) / max_rank),
        }
        for row in rows
    ]


async def retrieve_relevant_chunks(
    query: str,
    top_k: int | None = None,
    threshold: float | None = None,
    filter_metadata: dict[str, Any] | None = None,
    conversation_context: list[str] | None = None,
    search_mode: SearchMode = "hybrid",
    query_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve the most relevant document chunks for a query.

    Supports vector (cosine), keyword (full-text/BM25), and hybrid (RRF merge).

    Args:
        query: The search query text.
        top_k: Maximum number of results to return.
        threshold: Minimum similarity threshold (0-1). Only used for vector search.
        filter_metadata: Optional metadata filters.
        conversation_context: Optional recent user queries for multi-turn context.
        search_mode: One of "vector", "keyword", "hybrid".

    Returns:
        List of matching chunks with content, metadata, and similarity scores.
    """
    search_query = query
    settings = get_settings()
    top_k = top_k or settings.retrieval_top_k
    threshold = threshold if threshold is not None else settings.retrieval_threshold

    try:
        if search_mode == "vector":
            chunks = await _retrieve_vector(
                search_query=search_query,
                top_k=top_k,
                threshold=threshold,
                filter_metadata=filter_metadata,
                query_embedding=query_embedding,
            )
        elif search_mode == "keyword":
            chunks = await _retrieve_keyword(
                query=query,
                top_k=top_k,
                filter_metadata=filter_metadata,
            )
        elif search_mode == "hybrid":
            # Same params as vector for fair comparison (threshold, top_k per source)
            # Run vector and keyword retrieval in parallel to reduce latency
            vector_chunks, keyword_chunks = await asyncio.gather(
                _retrieve_vector(
                    search_query=search_query,
                    top_k=top_k,
                    threshold=threshold,
                    filter_metadata=filter_metadata,
                    query_embedding=query_embedding,
                ),
                _retrieve_keyword(
                    query=query,
                    top_k=top_k,
                    filter_metadata=filter_metadata,
                ),
            )

            vec_list = [(str(c["id"]), c["similarity"]) for c in vector_chunks]
            kw_list = [(str(c["id"]), c["similarity"]) for c in keyword_chunks]

            merged = rrf_merge([vec_list, kw_list])
            merged_ids = [cid for cid, _ in merged[:top_k]]
            id_to_chunk = {str(c["id"]): c for c in vector_chunks + keyword_chunks}

            chunks = []
            for cid in merged_ids:
                if cid in id_to_chunk:
                    chunks.append(id_to_chunk[cid])
                if len(chunks) >= top_k:
                    break
        else:
            raise ValueError(f"Unknown search_mode: {search_mode}")
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return []

    if not chunks:
        logger.warning(f"No results found for query: {query[:50]}...")
        return []

    formatted = [
        {
            "chunk_id": str(c["id"]),
            "content": c["content"],
            "similarity": c["similarity"],
            "metadata": c.get("metadata", {}),
        }
        for c in chunks
    ]

    logger.info(
        f"Retrieved {len(formatted)} chunks for query: {query[:50]}... "
        f"(mode={search_mode})"
    )
    return formatted


async def get_financials_chunks_for_pairs(
    ticker_date_pairs: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Fetch financial summary chunks for specific (ticker, call_date) pairs.

    Used to guarantee financials chunks are always in context for any call date
    already represented in the main retrieval results.
    """
    if not ticker_date_pairs:
        return []

    Chunk = _get_chunk_model()
    conditions = [
        and_(Chunk.company_ticker == t, Chunk.call_date == d)
        for t, d in ticker_date_pairs
    ]
    stmt = (
        select(Chunk.id, Chunk.content, Chunk.chunk_metadata, Chunk.company_ticker, Chunk.call_date)
        .where(or_(*conditions))
        .where(Chunk.chunk_metadata["chunk_type"].astext == "financials")
        .order_by(Chunk.company_ticker, Chunk.call_date.desc())
    )

    async with get_session() as session:
        result = await session.execute(stmt)
        rows = result.mappings().all()

    return [
        {
            "chunk_id": str(row["id"]),
            "content": row["content"],
            "similarity": 1.0,
            "metadata": row["chunk_metadata"] or {},
        }
        for row in rows
    ]


async def list_available_transcripts(
    company_ticker: str | None = None,
) -> list[dict[str, Any]]:
    """List distinct (company_ticker, call_date) available in the knowledge base.

    Args:
        company_ticker: Optional filter to scope to one company.

    Returns:
        List of dicts with company_ticker, call_date, title (from metadata).
    """
    Chunk = _get_chunk_model()
    title_col = Chunk.chunk_metadata["title"].astext

    stmt = (
        select(
            Chunk.company_ticker.label("company_ticker"),
            Chunk.call_date.label("call_date"),
            func.max(title_col).label("title"),
        )
        .where(Chunk.company_ticker.isnot(None))
        .where(Chunk.call_date.isnot(None))
        .group_by(Chunk.company_ticker, Chunk.call_date)
        .order_by(desc(Chunk.call_date))
    )
    if company_ticker:
        stmt = stmt.where(Chunk.company_ticker == company_ticker)

    async with get_session() as session:
        result = await session.execute(stmt)
        rows = result.mappings().all()

    return [
        {
            "company_ticker": row["company_ticker"],
            "call_date": row["call_date"],
            "title": row["title"] or f"{row['company_ticker']} {row['call_date']}",
        }
        for row in rows
    ]


# Process-local in-memory caches. Cleared explicitly on ingest.
# On multi-instance deployments, each process has its own cache; no distributed invalidation.
_COMPANIES_CACHE: dict = {}
_PERIODS_CACHE: dict = {}


async def get_known_companies() -> list[dict[str, str]]:
    """Query distinct company tickers and names from the knowledge base.

    Cached for 24 hours (process-local) since companies change only on ingestion.

    Returns:
        Sorted list of dicts with ``ticker`` and ``name`` keys,
        e.g. ``[{"ticker": "ACME", "name": "Acme Corp"}, ...]``.
    """
    cache_key = "companies_eval" if use_eval_chunks() else "companies"
    if cache_key in _COMPANIES_CACHE:
        return _COMPANIES_CACHE[cache_key]

    result = await _get_known_companies_impl()
    _COMPANIES_CACHE[cache_key] = result
    return result


async def _get_known_companies_impl() -> list[dict[str, str]]:
    """Internal: query companies from DB (uncached)."""
    Chunk = _get_chunk_model()
    title_col = Chunk.chunk_metadata["title"].astext

    stmt = (
        select(
            Chunk.company_ticker.label("company_ticker"),
            func.max(title_col).label("title"),
        )
        .where(Chunk.company_ticker.isnot(None))
        .group_by(Chunk.company_ticker)
        .order_by(Chunk.company_ticker)
    )

    async with get_session() as session:
        result = await session.execute(stmt)
        rows = result.mappings().all()

    companies: list[dict[str, str]] = []
    for row in rows:
        ticker = row["company_ticker"]
        name = TICKER_NAMES.get(ticker)
        if not name:
            title = row["title"] or ""
            name = re.sub(
                r"\s+(?:Q\d\s+\d{4}|Manual)\s+Earnings.*$", "", title, flags=re.IGNORECASE
            ).strip() or ticker
        companies.append({"ticker": ticker, "name": name})

    return companies


async def get_available_periods() -> dict[str, list[dict[str, Any]]]:
    """Return available periods per ticker: {ticker: [{call_date, fiscal_quarter, period_end}, ...]}.

    Each ticker's list is sorted by period_end descending (most recent first).
    Cached for 24h, cleared on ingestion.
    """
    cache_key = "periods_eval" if use_eval_chunks() else "periods"
    if cache_key in _PERIODS_CACHE:
        return _PERIODS_CACHE[cache_key]

    result = await _get_available_periods_impl()
    _PERIODS_CACHE[cache_key] = result
    return result


async def _get_available_periods_impl() -> dict[str, list[dict[str, Any]]]:
    """Internal: query available periods from DB (uncached)."""
    Chunk = _get_chunk_model()

    stmt = (
        select(
            Chunk.company_ticker,
            Chunk.call_date,
            Chunk.fiscal_quarter,
            Chunk.period_end,
        )
        .where(Chunk.company_ticker.isnot(None))
        .where(Chunk.call_date.isnot(None))
        .where(Chunk.period_end.isnot(None))
        .distinct()
        .order_by(Chunk.company_ticker, Chunk.period_end.desc())
    )

    async with get_session() as session:
        result = await session.execute(stmt)
        rows = result.all()

    periods: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        ticker = row[0]
        entry = {
            "call_date": row[1],
            "fiscal_quarter": row[2],
            "period_end": row[3].isoformat() if row[3] else None,
        }
        periods.setdefault(ticker, []).append(entry)

    return periods


async def retrieve_by_doc_id(doc_id: str, use_eval_table: bool = False) -> list[dict[str, Any]]:
    """Retrieve all chunks for a specific document.

    Args:
        doc_id: The source document ID.
        use_eval_table: If True, query eval_document_chunks instead of document_chunks.

    Returns:
        List of chunks ordered by chunk_index.
    """
    Chunk = EvalDocumentChunk if use_eval_table else DocumentChunk
    async with get_session() as session:
        result = await session.execute(
            select(Chunk)
            .where(Chunk.source_doc_id == doc_id)
            .order_by(Chunk.chunk_index)
        )
        chunks = result.scalars().all()

    return [
        {
            "id": str(c.id),
            "content": c.content,
            "metadata": c.chunk_metadata or {},
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]


async def get_transcript_by_chunk_id(chunk_id: str) -> dict[str, Any] | None:
    """Get the full transcript (all chunks) for the document containing the given chunk.

    Looks up the chunk in document_chunks first, then eval_document_chunks, so it
    works whether the source came from prod or eval retrieval.

    Args:
        chunk_id: UUID of any chunk belonging to the document.

    Returns:
        Dict with doc_id, title, metadata, chunks (ordered by chunk_index),
        requested_chunk_id, and optionally full_transcript (overlap-free) if stored.
        Returns None if chunk_id is not found.
    """
    try:
        uid = UUID(chunk_id)
    except (ValueError, TypeError):
        return None

    use_eval_table = False
    async with get_session() as session:
        result = await session.execute(
            select(DocumentChunk).where(DocumentChunk.id == uid)
        )
        chunk_row = result.scalar_one_or_none()
        if not chunk_row:
            result = await session.execute(
                select(EvalDocumentChunk).where(EvalDocumentChunk.id == uid)
            )
            chunk_row = result.scalar_one_or_none()
            if chunk_row:
                use_eval_table = True

    if not chunk_row or not chunk_row.source_doc_id:
        return None

    doc_id = chunk_row.source_doc_id
    chunks = await retrieve_by_doc_id(doc_id, use_eval_table=use_eval_table)
    if not chunks:
        return None

    first_meta = chunks[0].get("metadata") or {}
    title = first_meta.get("title") or first_meta.get("company_ticker") or doc_id
    # full_transcript avoids chunk overlap when available (stored in first chunk metadata)
    full_transcript = first_meta.get("_full_transcript")
    # Omit internal keys from metadata sent to frontend
    meta_out = {k: v for k, v in first_meta.items() if not k.startswith("_")}

    return {
        "doc_id": doc_id,
        "title": title,
        "metadata": meta_out,
        "chunks": chunks,
        "requested_chunk_id": chunk_id,
        "full_transcript": full_transcript,
    }
