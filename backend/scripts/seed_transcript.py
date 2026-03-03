"""Seed earnings transcripts into the knowledge base for evals.

Loads all *_transcript.json files from eval_data/ (or a specific file via --file).

    cd backend && python scripts/seed_transcript.py                    # all transcripts -> eval_document_chunks (default)
    cd backend && python scripts/seed_transcript.py --prod             # ingest into document_chunks (production)
    cd backend && python scripts/seed_transcript.py --file eval_data/acme_q3_2024_transcript.json
    cd backend && python scripts/run_evals.py
    cd backend && python scripts/run_retrieval_evals.py
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure backend is on path when run as script (e.g. python scripts/seed_transcript.py)
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.rag.ingestion import ingest_documents

# Transcript JSON files: backend/eval_data/
DATA_DIR = _BACKEND_ROOT / "eval_data"


def _parse_date(v):
    if v is None:
        return None
    if hasattr(v, "isoformat"):
        return v
    if isinstance(v, str):
        return datetime.fromisoformat(v.replace("Z", "+00:00")).date()
    return None


def _load_transcript_file(path: Path) -> list[dict]:
    """Load documents from a single transcript JSON file."""
    with open(path) as f:
        data = json.load(f)

    docs = []
    for d in data.get("documents", []):
        doc = dict(d)
        if isinstance(doc.get("call_date"), str):
            doc["call_date"] = _parse_date(doc["call_date"])
        docs.append(doc)
    return docs


def _discover_transcript_files() -> list[Path]:
    """Find all transcript JSON files in eval_data/."""
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.glob("*transcript*.json"))


async def main():
    """Load and ingest transcripts."""
    parser = argparse.ArgumentParser(description="Seed earnings transcripts")
    parser.add_argument(
        "--file",
        type=str,
        help="Path to a specific transcript JSON file (default: all in eval_data/)",
    )
    parser.add_argument(
        "--prod",
        action="store_true",
        help="Ingest into document_chunks (production). Default is eval_document_chunks.",
    )
    args = parser.parse_args()

    use_eval_table = not args.prod

    if args.file:
        p = Path(args.file)
        if not p.is_absolute() and not p.exists():
            p = _BACKEND_ROOT / p  # resolve relative to backend/
        files = [p]
        if not files[0].exists():
            print(f"Not found: {files[0]}")
            sys.exit(1)
    else:
        files = _discover_transcript_files()
        if not files:
            print(f"No transcript files found in {DATA_DIR}")
            sys.exit(1)

    print(f"Found {len(files)} transcript file(s):")
    for f in files:
        print(f"  - {f.name}")

    docs = []
    for f in files:
        loaded = _load_transcript_file(f)
        for d in loaded:
            d["use_eval_table"] = use_eval_table
        docs.extend(loaded)

    table = "eval_document_chunks" if use_eval_table else "document_chunks"
    print(f"\nIngesting {len(docs)} transcript(s) into {table}...")
    results = await ingest_documents(docs)

    for result in results:
        status = result["status"]
        title = result["title"]
        chunks = result["chunks_created"]
        print(f"  [{status}] {title} -> {chunks} chunks")

    total = sum(r["chunks_created"] for r in results)
    print(f"\nDone! Total chunks: {total}")
    print("Run evals with: python scripts/run_evals.py  (or python scripts/run_retrieval_evals.py)")


if __name__ == "__main__":
    asyncio.run(main())
