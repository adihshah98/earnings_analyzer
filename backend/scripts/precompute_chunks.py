"""Phase 1: Chunk transcripts, run LLM financials extraction, generate embeddings.

Saves results to backend/chunks/{TICKER}/{YYYY-MM-DD}.json.
No DB writes — safe to rerun. Run db_ingest_chunks.py to commit to DB.

Usage:
    cd backend && .venv/bin/python3 scripts/precompute_chunks.py               # all tickers
    cd backend && .venv/bin/python3 scripts/precompute_chunks.py --tickers IOT DDOG
    cd backend && .venv/bin/python3 scripts/precompute_chunks.py --force       # recompute even if .json exists
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_ROOT))

from dotenv import load_dotenv
load_dotenv(BACKEND_ROOT / ".env")

from app.ingestion.docx_parser import extract_text_from_docx
from app.rag.ingestion import (
    _generate_financials_chunk,
    chunk_transcript,
    generate_doc_id,
)
from app.rag.embeddings import generate_embeddings_batch
from app.rag.fiscal_calendar import compute_period_end
from app.rag.ingestion import _parse_fq_from_transcript_header, _infer_fq_from_call_date
from app.rag.ticker_map import TICKER_NAMES

TRANSCRIPTS_DIR = BACKEND_ROOT / "transcripts"
CHUNKS_DIR = BACKEND_ROOT / "chunks"


def _all_tickers() -> list[str]:
    return sorted(d.name for d in TRANSCRIPTS_DIR.iterdir() if d.is_dir())


async def precompute_file(docx_path: Path, ticker: str, force: bool) -> dict:
    call_date = docx_path.stem
    out_path = CHUNKS_DIR / ticker / f"{call_date}.json"

    if out_path.exists() and not force:
        print(f"  [SKIP] {ticker}/{call_date} — already computed (use --force to rerun)")
        return {"ticker": ticker, "call_date": call_date, "status": "skipped"}

    print(f"  Processing {ticker}/{call_date}...")

    with open(docx_path, "rb") as f:
        raw = f.read()
    text = extract_text_from_docx(raw)

    title = f"{ticker} Earnings Call {call_date}"
    company_name = TICKER_NAMES.get(ticker)
    from datetime import date as date_type
    call_date_obj = date_type.fromisoformat(call_date)
    doc_id = generate_doc_id(ticker, call_date_obj, title, "transcript")

    # Step 1: chunk
    chunk_items = chunk_transcript(text)
    chunk_contents = [c["content"] for c in chunk_items]

    # Steps 2+3: LLM financials + chunk embeddings in parallel
    (financials_content, llm_fiscal_quarter), chunk_embeddings = await asyncio.gather(
        _generate_financials_chunk(text, ticker, call_date, title),
        generate_embeddings_batch(chunk_contents),
    )

    # Financials embedding (1 vector, after LLM)
    financials_embedding = None
    if financials_content:
        financials_embedding = (await generate_embeddings_batch([financials_content]))[0]

    # Build output
    chunks_out = []
    for item, embedding in zip(chunk_items, chunk_embeddings):
        chunks_out.append({
            "content": item["content"],
            "speaker": item.get("speaker"),
            "speaker_role": item.get("speaker_role"),
            "embedding": embedding,
        })

    # Fiscal quarter: header first, LLM fallback, then infer from call_date
    fiscal_quarter = (
        _parse_fq_from_transcript_header(text)
        or llm_fiscal_quarter
        or _infer_fq_from_call_date(call_date_obj, ticker)
    )

    # Compute calendar period_end from fiscal_quarter + ticker's fiscal calendar
    period_end = compute_period_end(fiscal_quarter, ticker) if fiscal_quarter else None

    payload = {
        "ticker": ticker,
        "call_date": call_date,
        "doc_id": doc_id,
        "title": title,
        "fiscal_quarter": fiscal_quarter,
        "period_end": period_end.isoformat() if period_end else None,
        "company_name": company_name,
        "full_transcript": text,
        "chunks": chunks_out,
        "financials": {
            "content": financials_content,
            "embedding": financials_embedding,
        } if financials_content else None,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print(f"  [OK] {ticker}/{call_date} — {len(chunks_out)} chunks, FQ={fiscal_quarter or '?'} → {out_path.relative_to(BACKEND_ROOT)}")
    return {"ticker": ticker, "call_date": call_date, "status": "ok", "chunks": len(chunks_out)}


async def main(tickers: list[str], force: bool) -> None:
    tasks = []
    for ticker in tickers:
        ticker_dir = TRANSCRIPTS_DIR / ticker
        if not ticker_dir.exists():
            print(f"[SKIP] {ticker}: no transcript directory found")
            continue
        for docx_path in sorted(ticker_dir.glob("*.docx")):
            tasks.append((docx_path, ticker))

    print(f"Precomputing {len(tasks)} transcript(s) across {len(tickers)} ticker(s)...")
    print(f"Output dir: {CHUNKS_DIR}/\n")

    results = []
    for docx_path, ticker in tasks:
        result = await precompute_file(docx_path, ticker, force)
        results.append(result)

    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] == "error")
    print(f"\nDone. {ok} computed, {skipped} skipped, {failed} failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", help="Specific tickers (default: all)")
    parser.add_argument("--force", action="store_true", help="Recompute even if .json already exists")
    args = parser.parse_args()

    tickers = args.tickers if args.tickers else _all_tickers()
    asyncio.run(main(tickers, args.force))
