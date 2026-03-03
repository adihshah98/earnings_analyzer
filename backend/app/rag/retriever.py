"""pgvector retrieval for RAG pipeline via SQLAlchemy + asyncpg."""

import asyncio
import logging
import re
import time
from uuid import UUID

from typing import Any, Literal

from cachetools import TTLCache
from sqlalchemy import and_, desc, func, or_, select, text

from app.config import get_settings
from app.evals.context import use_eval_chunks
from app.models.database import get_session
from app.models.db_models import DocumentChunk, EvalDocumentChunk
from app.rag.embeddings import generate_embedding


def _get_chunk_model():
    """Return DocumentChunk or EvalDocumentChunk based on eval context."""
    return EvalDocumentChunk if use_eval_chunks() else DocumentChunk

logger = logging.getLogger(__name__)

SearchMode = Literal["vector", "keyword", "hybrid"]
RRF_K = 60  # Reciprocal Rank Fusion constant

# Stopwords to drop for OR-based keyword search (improves recall on natural-language queries)
_KEYWORD_STOPWORDS = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "has", "have", "this", "that",
    "with", "from", "what", "when", "where", "which", "who", "will",
    "your", "how", "many", "did", "in", "its", "into", "just", "than",
    "then", "them", "they", "been", "being", "would", "could", "should",
})


def _build_or_tsquery_terms(query: str, max_terms: int = 10) -> str | None:
    """Extract significant terms and build 'term1 | term2 | ...' for to_tsquery.

    Uses OR logic so chunks matching ANY term are returned (higher recall).
    ts_rank still favors chunks with more matches.
    """
    words = re.findall(r"\b[a-zA-Z0-9]+\b", query.lower())
    terms = [w for w in words if len(w) >= 2 and w not in _KEYWORD_STOPWORDS]
    if not terms:
        return None
    seen: set[str] = set()
    unique: list[str] = []
    for t in terms[:max_terms]:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return " | ".join(unique) if unique else None


def _build_search_query(query: str, conversation_context: list[str] | None = None) -> str:
    """Build search query with optional conversation context for better retrieval."""
    if not conversation_context:
        return query
    context_str = " ".join(conversation_context[-3:])  # Last 3 queries
    return f"Previous context: {context_str}. Current question: {query}".strip()


def _rrf_merge(
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


async def _resolve_as_of_date(
    as_of_date: str,
    company_ticker: str | None = None,
) -> list[tuple[str, str]]:
    """Resolve an as-of date to the latest call_date per company on or before that date.

    Because companies have different fiscal-year calendars, we normalise temporal
    queries by actual earnings-call date rather than fiscal quarter labels.

    Returns:
        List of (company_ticker, call_date) pairs — one per company that has at
        least one call on or before *as_of_date*.
    """
    Chunk = _get_chunk_model()
    ticker_col = Chunk.chunk_metadata["company_ticker"].astext
    date_col = Chunk.chunk_metadata["call_date"].astext

    stmt = (
        select(
            ticker_col.label("company_ticker"),
            func.max(date_col).label("call_date"),
        )
        .where(Chunk.chunk_metadata["call_date"].isnot(None))
        .where(date_col <= as_of_date)
        .group_by(ticker_col)
    )
    if company_ticker:
        stmt = stmt.where(ticker_col == company_ticker)

    async with get_session() as session:
        result = await session.execute(stmt)
        rows = result.mappings().all()

    resolved = [(row["company_ticker"], row["call_date"]) for row in rows]
    logger.info(
        f"Resolved as_of_date={as_of_date} (ticker={company_ticker}) → {resolved}"
    )
    return resolved


async def _resolve_entity_dates_batch(
    entity_dates: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Resolve all (ticker, as_of_date_iso) pairs in a single DB query.

    Uses the latest call on or before as_of_date per company. Returns only pairs
    that have a matching call in the DB.

    Args:
        entity_dates: List of (company_ticker, as_of_date_iso).

    Returns:
        List of (company_ticker, call_date) pairs.
    """
    if not entity_dates:
        return []

    Chunk = _get_chunk_model()
    table_name = Chunk.__table__.name

    tickers = [ticker for ticker, _ in entity_dates]
    as_of_dates = [as_of for _, as_of in entity_dates]

    t0 = time.perf_counter()
    # Single query: unnest + LATERAL to resolve all pairs (column is "metadata" in DB)
    stmt = text(f"""
        WITH inputs(ticker, as_of_date) AS (
            SELECT * FROM unnest(CAST(:tickers AS text[]), CAST(:as_of_dates AS text[])) AS t(ticker, as_of_date)
        )
        SELECT i.ticker, l.call_date
        FROM inputs i,
        LATERAL (
            SELECT MAX((c.metadata->>'call_date')) AS call_date
            FROM {table_name} c
            WHERE c.metadata->>'company_ticker' = i.ticker
              AND c.metadata->>'call_date' <= i.as_of_date
              AND c.metadata->>'call_date' IS NOT NULL
        ) l
        WHERE l.call_date IS NOT NULL
    """)

    async with get_session() as session:
        result = await session.execute(
            stmt, {"tickers": tickers, "as_of_dates": as_of_dates}
        )
        rows = result.mappings().all()

    pairs = [(row["ticker"], row["call_date"]) for row in rows]
    logger.info(
        "[latency] _resolve_entity_dates_batch: %.3fs (%d entities → %d pairs)",
        time.perf_counter() - t0,
        len(entity_dates),
        len(pairs),
    )
    return pairs


async def resolve_entities_to_call_dates(
    entity_dates: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Resolve (ticker, as_of_date_iso) to (ticker, call_date) for each entity.

    Uses a single batched query for all pairs to minimize DB round-trips.

    Args:
        entity_dates: List of (company_ticker, as_of_date_iso).

    Returns:
        List of (company_ticker, call_date) pairs.
    """
    return await _resolve_entity_dates_batch(entity_dates)


async def _build_resolved_filters(
    filter_metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Pre-process filter_metadata: resolve ``as_of_date`` into concrete call_date pairs.

    Returns a *new* dict (never mutates the input).
    """
    if not filter_metadata or "as_of_date" not in filter_metadata:
        return filter_metadata

    filters = dict(filter_metadata)
    as_of_date = str(filters.pop("as_of_date"))
    company_ticker = filters.get("company_ticker")

    resolved = await _resolve_as_of_date(as_of_date, company_ticker)
    if not resolved:
        filters["call_date"] = "__no_match__"
        return filters

    filters["_resolved_date_pairs"] = resolved
    return filters


def _apply_metadata_filters(stmt, filter_metadata: dict[str, Any] | None, chunk_model=None):
    """Apply JSONB metadata filters to a SQLAlchemy select statement.

    Handles the special ``_resolved_date_pairs`` key produced by
    ``_build_resolved_filters`` (from an ``as_of_date`` input).
    """
    Chunk = chunk_model or _get_chunk_model()
    if not filter_metadata:
        return stmt

    resolved_pairs = filter_metadata.get("_resolved_date_pairs")
    if resolved_pairs:
        ticker_col = Chunk.chunk_metadata["company_ticker"].astext
        date_col = Chunk.chunk_metadata["call_date"].astext
        conditions = [
            and_(ticker_col == ticker, date_col == call_date)
            for ticker, call_date in resolved_pairs
        ]
        stmt = stmt.where(or_(*conditions))

    for key, value in filter_metadata.items():
        if key.startswith("_"):
            continue
        if key == "company_ticker" and resolved_pairs:
            continue
        stmt = stmt.where(
            Chunk.chunk_metadata[key].astext == str(value)
        )
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
    """Retrieve chunks using PostgreSQL full-text search (ts_rank, BM25-style).

    Uses OR between significant terms (term1 | term2 | term3) so chunks matching
    ANY term are returned. ts_rank favors chunks with more matches. Fallback
    to plainto_tsquery if OR query fails or yields no terms.
    """
    Chunk = _get_chunk_model()
    or_str = _build_or_tsquery_terms(query)
    if or_str:
        ts_query = func.to_tsquery("english", or_str)
    else:
        ts_query = func.plainto_tsquery("english", query)
    rank_expr = func.coalesce(
        func.ts_rank(Chunk.content_tsv, ts_query), 0
    )

    stmt = (
        select(
            Chunk.id,
            Chunk.content,
            Chunk.chunk_metadata,
            rank_expr.label("similarity"),
        )
        .where(Chunk.content_tsv.op("@@")(ts_query))
        .order_by(desc(rank_expr))
        .limit(top_k)
    )
    stmt = _apply_metadata_filters(stmt, filter_metadata, Chunk)

    t0 = time.perf_counter()
    async with get_session() as session:
        try:
            result = await session.execute(stmt)
        except Exception as e:
            err = str(e).lower()
            if "content_tsv" in err:
                # Column doesn't exist (migration not run) - use to_tsvector on the fly
                tsv = func.to_tsvector("english", Chunk.content)
                ts_query_fb = func.plainto_tsquery("english", query)
                rank_expr_fb = func.coalesce(func.ts_rank(tsv, ts_query_fb), 0)
                stmt_fb = (
                    select(
                        Chunk.id,
                        Chunk.content,
                        Chunk.chunk_metadata,
                        rank_expr_fb.label("similarity"),
                    )
                    .where(tsv.op("@@")(ts_query_fb))
                    .order_by(desc(rank_expr_fb))
                    .limit(top_k)
                )
                stmt_fb = _apply_metadata_filters(stmt_fb, filter_metadata, Chunk)
                result = await session.execute(stmt_fb)
            elif "tsquery" in err or "syntax" in err:
                # to_tsquery failed (invalid OR syntax) - retry with plainto_tsquery
                ts_query_fb = func.plainto_tsquery("english", query)
                rank_expr_fb = func.coalesce(
                    func.ts_rank(Chunk.content_tsv, ts_query_fb), 0
                )
                stmt_fb = (
                    select(
                        Chunk.id,
                        Chunk.content,
                        Chunk.chunk_metadata,
                        rank_expr_fb.label("similarity"),
                    )
                    .where(Chunk.content_tsv.op("@@")(ts_query_fb))
                    .order_by(desc(rank_expr_fb))
                    .limit(top_k)
                )
                stmt_fb = _apply_metadata_filters(stmt_fb, filter_metadata, Chunk)
                result = await session.execute(stmt_fb)
            else:
                raise

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
    search_query = _build_search_query(query, conversation_context)
    settings = get_settings()
    top_k = top_k or settings.retrieval_top_k
    threshold = threshold if threshold is not None else settings.retrieval_threshold

    # Resolve as_of_date (if present) into concrete call_date filters.
    t0 = time.perf_counter()
    filter_metadata = await _build_resolved_filters(filter_metadata)
    logger.info("[latency] _build_resolved_filters: %.3fs", time.perf_counter() - t0)

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

            merged = _rrf_merge([vec_list, kw_list])
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
    ticker_col = Chunk.chunk_metadata["company_ticker"].astext
    date_col = Chunk.chunk_metadata["call_date"].astext
    title_col = Chunk.chunk_metadata["title"].astext

    stmt = (
        select(
            ticker_col.label("company_ticker"),
            date_col.label("call_date"),
            func.max(title_col).label("title"),
        )
        .where(Chunk.chunk_metadata["company_ticker"].isnot(None))
        .where(Chunk.chunk_metadata["call_date"].isnot(None))
        .group_by(ticker_col, date_col)
        .order_by(desc(date_col))
    )
    if company_ticker:
        stmt = stmt.where(ticker_col == company_ticker)

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


# Process-local cache for get_known_companies (24h TTL).
# On multi-instance deployments, each process has its own cache; no distributed cache.
_COMPANIES_CACHE: TTLCache = TTLCache(maxsize=2, ttl=86400)  # 24 hours


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
    ticker_col = Chunk.chunk_metadata["company_ticker"].astext
    title_col = Chunk.chunk_metadata["title"].astext

    stmt = (
        select(
            ticker_col.label("company_ticker"),
            func.max(title_col).label("title"),
        )
        .where(Chunk.chunk_metadata["company_ticker"].isnot(None))
        .group_by(ticker_col)
        .order_by(ticker_col)
    )

    async with get_session() as session:
        result = await session.execute(stmt)
        rows = result.mappings().all()

    companies: list[dict[str, str]] = []
    for row in rows:
        ticker = row["company_ticker"]
        title = row["title"] or ""
        name = re.sub(
            r"\s+Q\d\s+\d{4}\s+Earnings.*$", "", title, flags=re.IGNORECASE
        ).strip()
        if not name:
            name = ticker
        companies.append({"ticker": ticker, "name": name})

    return companies


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
