"""Re-extract financial summary chunks using gpt-4.1-mini for all ingested transcripts.

Reads the full transcript from the first chunk's metadata (_full_transcript),
re-runs financials extraction, and updates the existing financials chunk in-place.

Usage:
    cd backend && python scripts/reextract_financials.py          # all transcripts
    cd backend && python scripts/reextract_financials.py --ticker META  # single company
    cd backend && python scripts/reextract_financials.py --dry-run     # preview only
"""

import argparse
import asyncio
import logging
import re
import sys
import time
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from sqlalchemy import select, update

from app.models.database import get_session
from app.models.db_models import DocumentChunk
from app.rag.ingestion import _FINANCIALS_EXTRACTION_PROMPT
from app.rag.embeddings import generate_embeddings_batch, get_openai_client
from app.rag.fiscal_calendar import compute_period_end

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EXTRACTION_MODEL = "gpt-4.1-mini"

# Expected metrics to check for in each financials chunk
EXPECTED_METRICS = [
    "REVENUE",
    "GROSS MARGIN",
    "OPERATING",  # OpEx / Operating Income
    "FREE CASH FLOW",
    "CAPEX",
]

_BARE_FQ_RE = re.compile(r"^\s*(Q[1-4]\s*(?:FY|CY)?\s*\d{2,4})\s*$", re.IGNORECASE)


def _check_missing_metrics(content: str) -> list[str]:
    """Return list of expected metrics not found in the financials chunk content."""
    upper = content.upper()
    return [m for m in EXPECTED_METRICS if m not in upper]


async def _extract_financials(transcript: str, ticker: str, call_date: str, title: str) -> tuple[str | None, str | None]:
    """Extract financials using gpt-4.1-mini directly, with retries."""
    period_label = call_date or "unknown period"
    prompt = f"{_FINANCIALS_EXTRACTION_PROMPT}\n\nTranscript ({ticker}, {period_label}):\n{transcript}"

    client = get_openai_client()
    for attempt in range(5):
        try:
            response = await client.responses.create(
                model=EXTRACTION_MODEL,
                input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                max_output_tokens=600,
            )
            raw = (response.output_text if hasattr(response, "output_text") else "").strip()
            if not raw:
                return None, None

            # Parse fiscal quarter
            fiscal_quarter = None
            lines = raw.splitlines()
            metrics_lines = []
            for line in lines:
                if line.startswith("FISCAL_QUARTER:"):
                    fiscal_quarter = line.split(":", 1)[1].strip() or None
                elif not fiscal_quarter and _BARE_FQ_RE.match(line):
                    fiscal_quarter = _BARE_FQ_RE.match(line).group(1).strip()
                else:
                    metrics_lines.append(line)

            if fiscal_quarter:
                fiscal_quarter = re.sub(r"^(Q[1-4])\s+(\d{4})$", r"\1 FY\2", fiscal_quarter)

            period_label_full = f"{fiscal_quarter} | {call_date}" if fiscal_quarter else call_date or "unknown"
            header = f"FINANCIAL SUMMARY: {ticker} | {period_label_full}\n\n"
            metrics_body = "\n".join(metrics_lines).strip()
            if not metrics_body:
                return None, fiscal_quarter
            return header + metrics_body, fiscal_quarter

        except Exception as e:
            if "429" in str(e) and attempt < 4:
                wait = 10 * (attempt + 1)
                logger.info("  Rate limited, waiting %ds (attempt %d/5)", wait, attempt + 1)
                await asyncio.sleep(wait)
            else:
                raise
    return None, None


async def reextract_all(ticker_filter: str | None = None, dry_run: bool = False):
    """Re-extract financials for all transcripts (or a single ticker)."""
    logger.info("Using model: %s", EXTRACTION_MODEL)

    async with get_session() as session:
        stmt = (
            select(
                DocumentChunk.source_doc_id,
                DocumentChunk.company_ticker,
                DocumentChunk.call_date,
                DocumentChunk.chunk_metadata,
            )
            .where(DocumentChunk.chunk_index == 0)
            .order_by(DocumentChunk.company_ticker, DocumentChunk.call_date)
        )
        if ticker_filter:
            stmt = stmt.where(DocumentChunk.company_ticker == ticker_filter.upper())
        result = await session.execute(stmt)
        first_chunks = result.mappings().all()

    logger.info("Found %d transcripts to re-extract", len(first_chunks))

    missing_report: list[dict] = []
    updated = 0
    errors = 0

    for i, row in enumerate(first_chunks):
        doc_id = row["source_doc_id"]
        ticker = row["company_ticker"]
        call_date = row["call_date"]
        meta = row["chunk_metadata"] or {}
        transcript = meta.get("_full_transcript")
        title = meta.get("title", "")

        if not transcript:
            logger.warning("SKIP %s %s — no _full_transcript in metadata", ticker, call_date)
            errors += 1
            continue

        label = f"{ticker} {call_date}"

        if dry_run:
            logger.info("[dry-run] Would re-extract: %s (doc_id=%s)", label, doc_id)
            continue

        # Extract financials
        t0 = time.perf_counter()
        try:
            financials_content, fiscal_quarter = await _extract_financials(
                transcript, ticker, call_date, title
            )
        except Exception as e:
            logger.error("FAIL %s: %s", label, e)
            errors += 1
            continue
        elapsed = time.perf_counter() - t0

        if not financials_content:
            logger.warning("EMPTY %s — extraction returned no content (%.1fs)", label, elapsed)
            errors += 1
            continue

        # Check for missing metrics
        missing = _check_missing_metrics(financials_content)
        if missing:
            missing_report.append({
                "ticker": ticker,
                "call_date": call_date,
                "missing": missing,
            })

        # Compute new embedding
        new_embedding = (await generate_embeddings_batch([financials_content]))[0]

        # Compute period_end
        period_end_date = compute_period_end(fiscal_quarter, ticker) if fiscal_quarter else None

        # Update or insert
        async with get_session() as session:
            existing = await session.execute(
                select(DocumentChunk.id)
                .where(DocumentChunk.source_doc_id == doc_id)
                .where(DocumentChunk.chunk_metadata["chunk_type"].astext == "financials")
            )
            existing_row = existing.scalar_one_or_none()

            fin_meta = {
                "title": title,
                "source": meta.get("source", "transcript"),
                "company_ticker": ticker,
                "call_date": call_date,
                "chunk_type": "financials",
            }
            if fiscal_quarter:
                fin_meta["fiscal_quarter"] = fiscal_quarter
            if period_end_date:
                fin_meta["period_end"] = period_end_date.isoformat()

            if existing_row:
                await session.execute(
                    update(DocumentChunk)
                    .where(DocumentChunk.id == existing_row)
                    .values(
                        content=financials_content,
                        embedding=new_embedding,
                        chunk_metadata=fin_meta,
                        fiscal_quarter=fiscal_quarter,
                        period_end=period_end_date,
                    )
                )
            else:
                session.add(DocumentChunk(
                    content=financials_content,
                    chunk_metadata=fin_meta,
                    embedding=new_embedding,
                    source_doc_id=doc_id,
                    chunk_index=9999,
                    company_ticker=ticker,
                    call_date=call_date,
                    fiscal_quarter=fiscal_quarter,
                    period_end=period_end_date,
                ))

        updated += 1
        logger.info("[%d/%d] OK %s — fiscal_quarter=%s (%.1fs)%s",
                     i + 1, len(first_chunks), label, fiscal_quarter, elapsed,
                     f" [MISSING: {', '.join(missing)}]" if missing else "")

    # Final report
    logger.info("=" * 60)
    logger.info("Done. Updated: %d, Errors: %d, Total: %d", updated, errors, len(first_chunks))

    if missing_report:
        logger.info("")
        logger.info("MISSING METRICS REPORT:")
        logger.info("-" * 60)
        for entry in missing_report:
            logger.info("  %s %s — missing: %s",
                         entry["ticker"], entry["call_date"], ", ".join(entry["missing"]))
        logger.info("-" * 60)
        logger.info("Total with missing metrics: %d / %d", len(missing_report), updated)
    else:
        logger.info("All chunks have all expected metrics.")


def main():
    parser = argparse.ArgumentParser(description="Re-extract financial summary chunks")
    parser.add_argument("--ticker", help="Only re-extract for this ticker")
    parser.add_argument("--dry-run", action="store_true", help="Preview without updating")
    args = parser.parse_args()

    asyncio.run(reextract_all(ticker_filter=args.ticker, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
