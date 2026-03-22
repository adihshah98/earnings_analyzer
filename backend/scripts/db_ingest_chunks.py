"""Phase 2: Read precomputed chunks from backend/chunks/ and write to DB.

No LLM calls, no embeddings — just DB writes. Fast and safe to rerun (upserts by doc_id).
Run precompute_chunks.py first to generate the .json files.

Usage:
    cd backend && .venv/bin/python3 scripts/db_ingest_chunks.py               # all tickers
    cd backend && .venv/bin/python3 scripts/db_ingest_chunks.py --tickers IOT DDOG
    cd backend && .venv/bin/python3 scripts/db_ingest_chunks.py --eval        # write to eval table
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_ROOT))

from dotenv import load_dotenv
load_dotenv(BACKEND_ROOT / ".env")

from datetime import date as date_type
from sqlalchemy import delete
from app.models.database import get_session
from app.models.db_models import DocumentChunk, EvalDocumentChunk
from app.rag.fiscal_calendar import compute_period_end
from app.rag.retriever import _COMPANIES_CACHE, _PERIODS_CACHE

CHUNKS_DIR = BACKEND_ROOT / "chunks"


def _all_tickers() -> list[str]:
    if not CHUNKS_DIR.exists():
        return []
    return sorted(d.name for d in CHUNKS_DIR.iterdir() if d.is_dir())


async def ingest_file(json_path: Path, use_eval_table: bool) -> dict:
    Chunk = EvalDocumentChunk if use_eval_table else DocumentChunk

    with open(json_path, encoding="utf-8") as f:
        payload = json.load(f)

    ticker = payload["ticker"]
    call_date = payload["call_date"]
    doc_id = payload["doc_id"]
    title = payload["title"]
    fiscal_quarter = payload.get("fiscal_quarter")
    # Read period_end from payload (new precomputed files have it); compute as fallback
    period_end_raw = payload.get("period_end")
    if period_end_raw:
        period_end = date_type.fromisoformat(period_end_raw)
    else:
        period_end = compute_period_end(fiscal_quarter, ticker) if fiscal_quarter else None
    company_name = payload.get("company_name")
    full_transcript = payload.get("full_transcript", "")
    chunks = payload["chunks"]
    financials = payload.get("financials")

    base_metadata: dict[str, Any] = {
        "title": title,
        "source": "transcript",
        "company_ticker": ticker,
        "call_date": call_date,
        **({"fiscal_quarter": fiscal_quarter} if fiscal_quarter else {}),
        **({"period_end": period_end.isoformat()} if period_end else {}),
        **({"company_name": company_name} if company_name else {}),
    }

    async with get_session() as session:
        # Delete existing chunks for this document (upsert by doc_id)
        await session.execute(delete(Chunk).where(Chunk.source_doc_id == doc_id))

        for i, chunk in enumerate(chunks):
            meta = {**base_metadata}
            if chunk.get("speaker"):
                meta["speaker"] = chunk["speaker"]
            if chunk.get("speaker_role"):
                meta["speaker_role"] = chunk["speaker_role"]
            if i == 0:
                meta["_full_transcript"] = full_transcript
            session.add(Chunk(
                content=chunk["content"],
                chunk_metadata=meta,
                embedding=chunk["embedding"],
                source_doc_id=doc_id,
                chunk_index=i,
                company_ticker=ticker,
                call_date=call_date,
                fiscal_quarter=fiscal_quarter,
                period_end=period_end,
            ))

        if financials and financials.get("content") and financials.get("embedding"):
            fin_meta = {**base_metadata, "chunk_type": "financials"}
            session.add(Chunk(
                content=financials["content"],
                chunk_metadata=fin_meta,
                embedding=financials["embedding"],
                source_doc_id=doc_id,
                chunk_index=len(chunks),
                company_ticker=ticker,
                call_date=call_date,
                fiscal_quarter=fiscal_quarter,
                period_end=period_end,
            ))

    n_fin = 1 if (financials and financials.get("content")) else 0
    print(f"  [OK] {ticker}/{call_date} — {len(chunks)} chunks + {n_fin} financials chunk written (doc_id={doc_id})")
    return {"ticker": ticker, "call_date": call_date, "status": "ok"}


async def main(tickers: list[str], use_eval_table: bool) -> None:
    tasks = []
    for ticker in tickers:
        ticker_dir = CHUNKS_DIR / ticker
        if not ticker_dir.exists():
            print(f"[SKIP] {ticker}: no chunks directory found — run precompute_chunks.py first")
            continue
        for json_path in sorted(ticker_dir.glob("*.json")):
            tasks.append((json_path, ticker))

    if not tasks:
        print("No chunk files found. Run precompute_chunks.py first.")
        sys.exit(1)

    table = "eval_document_chunks" if use_eval_table else "document_chunks"
    print(f"Writing {len(tasks)} transcript(s) to {table}...\n")

    results = []
    for json_path, ticker in tasks:
        try:
            result = await ingest_file(json_path, use_eval_table)
        except Exception as e:
            print(f"  [FAIL] {ticker}/{json_path.stem}: {e}")
            result = {"ticker": ticker, "call_date": json_path.stem, "status": "error"}
        results.append(result)

    # Clear caches so new tickers/periods are immediately visible
    _COMPANIES_CACHE.clear()
    _PERIODS_CACHE.clear()

    ok = sum(1 for r in results if r["status"] == "ok")
    failed = sum(1 for r in results if r["status"] == "error")
    print(f"\nDone. {ok}/{len(tasks)} ingested successfully." + (f" {failed} failed." if failed else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", help="Specific tickers (default: all with precomputed chunks)")
    parser.add_argument("--eval", action="store_true", help="Write to eval_document_chunks instead")
    args = parser.parse_args()

    tickers = args.tickers if args.tickers else _all_tickers()
    asyncio.run(main(tickers, args.eval))
