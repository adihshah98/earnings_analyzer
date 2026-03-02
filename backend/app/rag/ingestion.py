"""Document chunking and ingestion pipeline."""

import hashlib
import logging
import re
from datetime import date
from typing import Any

import tiktoken
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.database import get_session
from app.models.db_models import DocumentChunk, EvalDocumentChunk
from app.rag.embeddings import generate_embeddings_batch

logger = logging.getLogger(__name__)


def _get_tokenizer():
    """Get tiktoken tokenizer for chunk size calculation."""
    return tiktoken.encoding_for_model("gpt-4o-mini")


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """Split text into overlapping chunks by token count.

    Args:
        text: The text to chunk.
        chunk_size: Max tokens per chunk.
        chunk_overlap: Number of overlapping tokens between chunks.

    Returns:
        List of text chunks.
    """
    if chunk_size is None or chunk_overlap is None:
        settings = get_settings()
        chunk_size = chunk_size or settings.chunk_size
        chunk_overlap = chunk_overlap or settings.chunk_overlap

    tokenizer = _get_tokenizer()
    tokens = tokenizer.encode(text)

    if len(tokens) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)

        start = end - chunk_overlap

    return chunks


def _parse_speaker_turn(line: str) -> tuple[str, str | None, str] | None:
    """Parse a line as 'Speaker [Role]: [content]'. Returns (speaker, role, content) or None.

    Handles: "Name -- Role: content", "Name, Role: content", "Name: content"
    (standalone "Operator:" is just content="").
    """
    stripped = line.strip()
    if not stripped or ":" not in stripped:
        return None
    # Name -- Role: content
    m = re.match(r"^(.+?)\s*[-–—]+\s*(.+?)\s*:\s*(.*)$", stripped)
    if m:
        return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    # Name, Role: content
    m = re.match(r"^(.+?)\s*,\s*(.+?)\s*:\s*(.*)$", stripped)
    if m:
        return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    # Name: content (prefix capped at 80 chars so we don't treat long prose as a speaker)
    m = re.match(r"^([^:]{1,80}):\s*(.*)$", stripped)
    if m:
        return m.group(1).strip(), None, m.group(2).strip()
    return None


def chunk_transcript(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict[str, Any]]:
    """Split transcript into chunks by speaker turns, preserving speaker metadata.

    Splits on speaker header lines (e.g. "Tim Cook -- CEO:" or "Operator:").
    Keeps each speaker block as a unit; only sub-chunks if a block exceeds chunk_size.

    Args:
        text: Transcript text.
        chunk_size: Max tokens per chunk.
        chunk_overlap: Overlap when sub-chunking large blocks.

    Returns:
        List of dicts with "content", "speaker", "speaker_role" keys.
    """
    if chunk_size is None or chunk_overlap is None:
        settings = get_settings()
        chunk_size = chunk_size or settings.chunk_size
        chunk_overlap = chunk_overlap or settings.chunk_overlap

    blocks: list[dict[str, Any]] = []
    current_speaker: str | None = None
    current_role: str | None = None
    current_lines: list[str] = []

    def flush_block() -> None:
        if current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                blocks.append({
                    "content": content,
                    "speaker": current_speaker,
                    "speaker_role": current_role,
                })

    lines = text.split("\n")
    for line in lines:
        stripped = line.strip()
        parsed = _parse_speaker_turn(stripped)
        if parsed:
            speaker, role, content = parsed
            flush_block()
            current_speaker = speaker
            current_role = role
            current_lines = [content] if content else []
        elif stripped:
            current_lines.append(line)

    flush_block()

    # No speaker structure found: treat as single block
    if not blocks:
        return [{"content": text, "speaker": None, "speaker_role": None}]

    # Sub-chunk blocks that exceed chunk_size
    tokenizer = _get_tokenizer()
    final: list[dict[str, Any]] = []
    for block in blocks:
        block_text = block["content"]
        tokens = tokenizer.encode(block_text)
        if len(tokens) <= chunk_size:
            final.append(block)
        else:
            sub_chunks = chunk_text(block_text, chunk_size, chunk_overlap)
            for sc in sub_chunks:
                final.append({
                    "content": sc,
                    "speaker": block["speaker"],
                    "speaker_role": block["speaker_role"],
                })
    return final


def generate_doc_id(company_ticker: str, call_date: date | None, title: str = "", source: str = "") -> str:
    """Generate a deterministic document ID for earnings transcripts."""
    if call_date:
        content = f"{company_ticker}:{call_date.isoformat()}"
    else:
        content = f"{company_ticker}:{title}:{source}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


async def ingest_document(
    title: str,
    content: str,
    source: str = "transcript",
    company_ticker: str | None = None,
    call_date: date | None = None,
    metadata: dict[str, Any] | None = None,
    use_transcript_chunking: bool = True,
    use_eval_table: bool = False,
) -> dict[str, Any]:
    """Ingest a document: chunk, embed, and store in PostgreSQL.

    Args:
        title: Document title.
        content: Full document text.
        source: Source identifier.
        company_ticker: Company ticker (e.g. AAPL). Required for earnings transcripts.
        call_date: Earnings call date. Source of truth for time.
        metadata: Additional metadata to store with chunks.
        use_transcript_chunking: Use speaker-turn chunking for transcripts.

    use_eval_table: If True, store in eval_document_chunks instead of document_chunks.

    Returns:
        Dict with doc_id, chunk count, and status.
    """
    Chunk = EvalDocumentChunk if use_eval_table else DocumentChunk
    metadata = metadata or {}
    ticker = company_ticker or metadata.get("company_ticker", "unknown")
    doc_id = generate_doc_id(ticker, call_date, title, source)

    # Chunk the document
    if use_transcript_chunking:
        chunk_items = chunk_transcript(content)
        chunk_contents = [c["content"] for c in chunk_items]
        chunk_speakers = [c.get("speaker") for c in chunk_items]
        chunk_roles = [c.get("speaker_role") for c in chunk_items]
    else:
        chunk_contents = chunk_text(content)
        chunk_speakers = [None] * len(chunk_contents)
        chunk_roles = [None] * len(chunk_contents)

    logger.info(f"Document '{title}' split into {len(chunk_contents)} chunks")

    if not chunk_contents:
        return {"doc_id": doc_id, "chunks_created": 0, "status": "empty"}

    # Generate embeddings for all chunks
    embeddings = await generate_embeddings_batch(chunk_contents)

    async with get_session() as session:
        # Delete existing chunks for this document
        await session.execute(
            delete(Chunk).where(
                Chunk.source_doc_id == doc_id,
            ),
        )

        # Insert new chunks
        for i, (chunk_content, embedding) in enumerate(zip(chunk_contents, embeddings)):
            base_meta: dict[str, Any] = {
                **metadata,
                "title": title,
                "source": source,
                "company_ticker": ticker,
                "call_date": str(call_date) if call_date else None,
            }
            if chunk_speakers[i]:
                base_meta["speaker"] = chunk_speakers[i]
            if chunk_roles[i]:
                base_meta["speaker_role"] = chunk_roles[i]
            # Store full transcript in first chunk for overlap-free display (chunks can overlap at boundaries)
            if i == 0:
                base_meta["_full_transcript"] = content
            chunk_metadata = base_meta
            row = Chunk(
                content=chunk_content,
                chunk_metadata=chunk_metadata,
                embedding=embedding,
                source_doc_id=doc_id,
                chunk_index=i,
            )
            session.add(row)

    logger.info(f"Ingested document '{title}' with {len(chunk_contents)} chunks (doc_id={doc_id})")

    return {
        "doc_id": doc_id,
        "title": title,
        "chunks_created": len(chunk_contents),
        "status": "success",
    }


async def ingest_documents(
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ingest multiple documents.

    Args:
        documents: List of dicts with title, content, source, company_ticker,
            call_date, metadata.

    Returns:
        List of ingestion results.
    """
    results = []
    for doc in documents:
        result = await ingest_document(
            title=doc["title"],
            content=doc["content"],
            source=doc.get("source", "transcript"),
            company_ticker=doc.get("company_ticker"),
            call_date=doc.get("call_date"),
            metadata=doc.get("metadata"),
            use_transcript_chunking=doc.get("use_transcript_chunking", True),
            use_eval_table=doc.get("use_eval_table", False),
        )
        results.append(result)
    return results
