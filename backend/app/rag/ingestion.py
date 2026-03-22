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
from app.rag.embeddings import generate_embeddings_batch, get_openai_client
from app.rag.fiscal_calendar import FY_END_MONTH, _compute_period_end_from_parts, compute_period_end

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


def _is_standalone_speaker_header(line: str) -> str | None:
    """If line looks like 'Name Company – Role' (no colon, has en/em dash), return it; else None.

    Used for transcripts where the speaker is on its own line, e.g.:
    "Darren Yip ServiceNow, Inc. – Head of Investor Relations"
    """
    stripped = line.strip()
    if not stripped or ":" in stripped or len(stripped) > 120:
        return None
    if re.search(r"\s+[–—]\s+", stripped):
        return stripped
    return None


def _split_line_on_mid_speaker(line: str) -> tuple[str, str, str] | None:
    """If line contains a speaker header after a sentence (e.g. '. Name Co – Role '), return (before, header, after); else None.

    Handles cases where the source has no newline before the speaker, e.g.:
    "...for our use cases as well. William R. McDermott ServiceNow, Inc. – Chairman & CEO And may I just..."
    Splits at the dash: header is "Name Company – Role"; role ends at a sentence boundary (space + capital + optional lowercase + space), not hardcoded words.
    """
    if " – " not in line and " — " not in line:
        return None
    # After a period and space, capture "Name ... Company – Role"; role ends when we see \s+ then a word (Capital + optional lowercase) + space (start of next sentence).
    m = re.match(
        r"(.*?[.])\s+([A-Z][^–—]*?\s+[–—]\s+[A-Za-z\s&\-]+?)(?=\s+[A-Z][a-z]*\s)\s+([A-Z].*)",
        line,
        re.DOTALL,
    )
    if not m:
        return None
    before, header, after = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    if len(header) > 120:
        return None
    return (before, header, after)


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

    Recognizes two formats:
    - Colon style: "Name – Role: content" or "Name: content" on one line.
    - Standalone header: "Name Company – Role" on its own line, content on following lines.

    Stored chunk content is formatted as "Speaker Line\\n\\nContent" for retrieval.
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

    lines = re.split(r"\n{2,}", text)
    for line in lines:
        stripped = line.strip()
        parsed = _parse_speaker_turn(stripped)
        if parsed:
            speaker, role, content = parsed
            flush_block()
            current_speaker = speaker
            current_role = role
            current_lines = [content] if content else []
        else:
            standalone = _is_standalone_speaker_header(stripped)
            if standalone is not None:
                flush_block()
                current_speaker = standalone
                current_role = None
                current_lines = []
            else:
                mid_split = _split_line_on_mid_speaker(line)
                if mid_split is not None:
                    before, header, after = mid_split
                    if before:
                        current_lines.append(before)
                    flush_block()
                    current_speaker = header
                    current_role = None
                    current_lines = [after] if after else []
                elif stripped:
                    current_lines.append(line)

    flush_block()

    # No speaker structure found: treat as single block
    if not blocks:
        return [{"content": text, "speaker": None, "speaker_role": None}]

    def _speaker_header(block: dict[str, Any]) -> str | None:
        if block.get("speaker") is None:
            return None
        if block.get("speaker_role"):
            return f"{block['speaker']} – {block['speaker_role']}"
        return block["speaker"]

    def _format_chunk_content(header: str | None, body: str) -> str:
        if header:
            return f"{header}\n\n{body}".strip()
        return body

    # Sub-chunk blocks that exceed chunk_size; format stored content as "Speaker Line\n\nContent"
    tokenizer = _get_tokenizer()
    final: list[dict[str, Any]] = []
    for block in blocks:
        body = block["content"]
        header = _speaker_header(block)
        tokens = tokenizer.encode(body)
        if len(tokens) <= chunk_size:
            final.append({
                "content": _format_chunk_content(header, body),
                "speaker": block["speaker"],
                "speaker_role": block["speaker_role"],
            })
        else:
            sub_chunks = chunk_text(body, chunk_size, chunk_overlap)
            for sc in sub_chunks:
                final.append({
                    "content": _format_chunk_content(header, sc),
                    "speaker": block["speaker"],
                    "speaker_role": block["speaker_role"],
                })
    return final


# Regex to extract fiscal quarter from the transcript header line, e.g.:
# "Alphabet Inc., Q3 2022 Earnings Call, Oct 25, 2022"
# "Salesforce, Inc., Q1 2025 Earnings Call, May 29, 2024"
_HEADER_FQ_RE = re.compile(r"(Q[1-4])\s+(\d{4})\s+Earnings\s+Call", re.IGNORECASE)


def _parse_fq_from_transcript_header(content: str) -> str | None:
    """Try to extract fiscal quarter from the first few lines of transcript text."""
    header = content[:500]
    m = _HEADER_FQ_RE.search(header)
    if m:
        q = m.group(1).upper()
        year = m.group(2)
        return f"{q} FY{year}"
    return None


def _infer_fq_from_call_date(call_date: date, ticker: str) -> str | None:
    """Infer fiscal quarter from call_date + ticker's fiscal calendar.

    Earnings calls happen ~1-2 months after the quarter ends. Try all
    (quarter, fy_year) combos nearby, pick the one whose period_end is
    closest before the call_date (within 90 days).
    """
    fy_end_month = FY_END_MONTH.get(ticker.upper(), 12)
    best: str | None = None
    best_gap = 91
    for fy_year in (call_date.year - 1, call_date.year, call_date.year + 1):
        for q in range(1, 5):
            period_end = _compute_period_end_from_parts(q, fy_year, fy_end_month)
            gap = (call_date - period_end).days
            if 0 <= gap < best_gap:
                best = f"Q{q} FY{fy_year}"
                best_gap = gap
    return best


_FINANCIALS_EXTRACTION_PROMPT = """\
You are extracting key financial metrics from an earnings call transcript.

For EVERY metric found, you must:
1. Label it as REPORTED (actual results for the period just completed) or GUIDED (forward-looking outlook)
2. Include the exact period (e.g. "Q1 FY2026", "FY2026 full year", "Q2 FY2026 outlook")
3. Include exact figures with units ($M/$B, %, count)

Metrics to extract (omit any not mentioned):
- Revenue (quarterly)
- Revenue (annual / full-year)
- NRR / Net Revenue Retention
- ACV / Annual Contract Value
- Customer Count
- Gross Margin
- Operating Expenses (OpEx) / Operating Income/Loss
- Free Cash Flow / Operating Cash Flow
- CapEx / Capital Expenditure
- cRPO / Current Remaining Performance Obligation
- Cash & Cash Equivalents / Cash Balance

Output format — first line is the fiscal quarter, then one line per metric:
FISCAL_QUARTER: Q1 FY2026
REVENUE (QUARTERLY): Reported $322.2M for Q1 FY2026 (+32% YoY); Guided $337–339M for Q2 FY2026
REVENUE (ANNUAL): Guided $1.355–1.360B for FY2026 full year
GROSS MARGIN: Reported 75.5% for Q1 FY2026; Guided ~75% for Q2 FY2026
NRR: 115% as of Q1 FY2026
CUSTOMER COUNT: 1,848 customers with ARR >$100K as of Q1 FY2026 (+21% YoY)
FREE CASH FLOW: Reported $47.2M for Q1 FY2026 (14.7% margin)
CASH BALANCE: $868.4M as of end of Q1 FY2026

Always output FISCAL_QUARTER on the first line. Look at the transcript heading/title (e.g. "Q3 2021 Earnings Call") or any early mention of the quarter to determine this — it does not need to be explicitly labeled "fiscal quarter" in the text. Output it in the format Q{N} FY{YYYY} (e.g. Q1 FY2026, Q3 FY2021).
Be concise — metrics and figures only, no narrative commentary.\
"""


async def _generate_financials_chunk(
    content: str,
    ticker: str,
    call_date_str: str | None,
    title: str,
) -> tuple[str | None, str | None]:
    """Extract a structured financial summary and fiscal quarter from transcript text via LLM.

    Returns (chunk_content, fiscal_quarter). Either may be None on failure or missing data.
    """
    period_label = call_date_str or "unknown period"
    prompt = f"{_FINANCIALS_EXTRACTION_PROMPT}\n\nTranscript ({ticker}, {period_label}):\n{content}"
    try:
        client = get_openai_client()
        response = await client.responses.create(
            model=get_settings().openai_model,
            input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            max_output_tokens=600,
        )
        raw = (response.output_text if hasattr(response, "output_text") else "").strip()
        if not raw:
            return None, None

        # Parse fiscal quarter — handles both "FISCAL_QUARTER: Q1 FY2026" and bare "Q1 FY2026"
        fiscal_quarter: str | None = None
        lines = raw.splitlines()
        metrics_lines = []
        _bare_fq_re = re.compile(r"^\s*(Q[1-4]\s*(?:FY|CY)?\s*\d{2,4})\s*$", re.IGNORECASE)
        for line in lines:
            if line.startswith("FISCAL_QUARTER:"):
                fiscal_quarter = line.split(":", 1)[1].strip() or None
            elif not fiscal_quarter and _bare_fq_re.match(line):
                fiscal_quarter = _bare_fq_re.match(line).group(1).strip()
            else:
                metrics_lines.append(line)

        period_label_full = f"{fiscal_quarter} | {call_date_str}" if fiscal_quarter else call_date_str or "unknown period"
        header = f"FINANCIAL SUMMARY: {ticker} | {period_label_full}\n\n"
        metrics_body = "\n".join(metrics_lines).strip()
        if not metrics_body:
            return None, fiscal_quarter
        return header + metrics_body, fiscal_quarter
    except Exception as e:
        logger.warning("Financials chunk extraction failed for %s %s: %s", ticker, period_label, e)
        return None, None


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

    call_date_str = str(call_date) if call_date else None

    # Run financials LLM extraction and chunk embeddings in parallel (independent).
    (financials_content, llm_fiscal_quarter), embeddings = await asyncio.gather(
        _generate_financials_chunk(content, ticker, call_date_str, title),
        generate_embeddings_batch(chunk_contents),
    )

    # Fiscal quarter extraction — deterministic first, LLM as fallback:
    # 1) Transcript header (e.g. "Q3 2022 Earnings Call") — always present, free
    # 2) LLM output from financials extraction — already computed above
    # 3) Infer from call_date + ticker's fiscal calendar — last resort
    fiscal_quarter = (
        _parse_fq_from_transcript_header(content)
        or llm_fiscal_quarter
        or (_infer_fq_from_call_date(call_date, ticker) if call_date else None)
    )

    # Compute period_end from fiscal_quarter + ticker's fiscal calendar.
    period_end_date = compute_period_end(fiscal_quarter, ticker) if fiscal_quarter else None

    # Embed the financials chunk separately (only 1 vector, done after LLM completes).
    if financials_content:
        financials_embedding = (await generate_embeddings_batch([financials_content]))[0]
    else:
        financials_embedding = None

    async with get_session() as session:
        # Delete existing chunks for this document
        await session.execute(
            delete(Chunk).where(
                Chunk.source_doc_id == doc_id,
            ),
        )

        # Insert main chunks
        for i, (chunk_content, embedding) in enumerate(zip(chunk_contents, embeddings)):
            base_meta: dict[str, Any] = {
                **metadata,
                "title": title,
                "source": source,
                "company_ticker": ticker,
                "call_date": call_date_str,
            }
            if fiscal_quarter:
                base_meta["fiscal_quarter"] = fiscal_quarter
            if period_end_date:
                base_meta["period_end"] = period_end_date.isoformat()
            if chunk_speakers[i]:
                base_meta["speaker"] = chunk_speakers[i]
            if chunk_roles[i]:
                base_meta["speaker_role"] = chunk_roles[i]
            # Store full transcript in first chunk for overlap-free display (chunks can overlap at boundaries)
            if i == 0:
                base_meta["_full_transcript"] = content
            row = Chunk(
                content=chunk_content,
                chunk_metadata=base_meta,
                embedding=embedding,
                source_doc_id=doc_id,
                chunk_index=i,
                company_ticker=ticker,
                call_date=call_date_str,
                fiscal_quarter=fiscal_quarter,
                period_end=period_end_date,
            )
            session.add(row)

        # Insert financial summary chunk
        if financials_content and financials_embedding is not None:
            fin_meta: dict[str, Any] = {
                **metadata,
                "title": title,
                "source": source,
                "company_ticker": ticker,
                "call_date": call_date_str,
                "chunk_type": "financials",
            }
            if fiscal_quarter:
                fin_meta["fiscal_quarter"] = fiscal_quarter
            if period_end_date:
                fin_meta["period_end"] = period_end_date.isoformat()
            session.add(Chunk(
                content=financials_content,
                chunk_metadata=fin_meta,
                embedding=financials_embedding,
                source_doc_id=doc_id,
                chunk_index=len(chunk_contents),
                company_ticker=ticker,
                call_date=call_date_str,
                fiscal_quarter=fiscal_quarter,
                period_end=period_end_date,
            ))

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
