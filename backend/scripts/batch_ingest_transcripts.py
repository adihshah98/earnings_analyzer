"""Batch ingest transcripts from a folder structure: transcripts/{ticker}/{date}.docx

Folder structure:
  backend/transcripts/
    AAPL/
      2024-10-30.docx
      2024-07-25.docx
    MSFT/
      2024-10-31.docx

Supports .docx and .txt. For .txt, converts to .docx before upload.
Filename (without extension) is parsed as the call date (YYYY-MM-DD or YYYYMMDD).

Usage:
  cd backend && python scripts/batch_ingest_transcripts.py
  cd backend && python scripts/batch_ingest_transcripts.py --dir ./my_transcripts --base-url http://localhost:8000
  cd backend && python scripts/batch_ingest_transcripts.py --dry-run
"""

import argparse
import io
import os
import re
import sys
from pathlib import Path

try:
    import httpx
except ImportError:
    print("ERROR: httpx required. pip install httpx")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from docx import Document
except ImportError:
    Document = None

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent
DEFAULT_TRANSCRIPTS_DIR = BACKEND_ROOT / "transcripts"
DEFAULT_BASE_URL = "https://earnings-analyzer-dud6.onrender.com"
SUPPORTED_EXTENSIONS = (".docx", ".txt")

# Load backend/.env
if load_dotenv:
    load_dotenv(BACKEND_ROOT / ".env")


def _parse_date_from_stem(stem: str) -> str | None:
    """Parse call date from filename stem. Returns YYYY-MM-DD or None.

    Supports: 2024-10-30, 20241030, 2024_10_30
    """
    stem = stem.strip()
    # YYYY-MM-DD
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", stem)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    # YYYYMMDD
    m = re.match(r"^(\d{4})(\d{2})(\d{2})$", stem)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    # YYYY_MM_DD
    m = re.match(r"^(\d{4})_(\d{2})_(\d{2})$", stem)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def _txt_to_docx_bytes(text: str) -> bytes:
    """Convert plain text to a minimal .docx in memory."""
    if Document is None:
        raise RuntimeError("python-docx required for .txt conversion. pip install python-docx")
    doc = Document()
    for para in text.split("\n\n"):
        doc.add_paragraph(para.strip())
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.read()


def _discover_files(transcripts_dir: Path) -> list[tuple[Path, str, str]]:
    """Yield (file_path, ticker, call_date) for each ingestible file."""
    if not transcripts_dir.exists():
        return []

    results: list[tuple[Path, str, str]] = []
    for ticker_dir in sorted(transcripts_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name.strip().upper()
        if not ticker:
            continue
        for f in sorted(ticker_dir.iterdir()):
            if not f.is_file():
                continue
            if f.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            stem = f.stem
            call_date = _parse_date_from_stem(stem)
            if not call_date:
                print(f"  [SKIP] {f.relative_to(transcripts_dir)}: cannot parse date from '{stem}'")
                continue
            results.append((f, ticker, call_date))
    return results


def upload_file(
    file_path: Path,
    ticker: str,
    call_date: str,
    base_url: str,
    admin_key: str,
    use_eval_table: bool = False,
) -> dict:
    """Upload a single file to the manual ingest API."""
    url = f"{base_url.rstrip('/')}/rag/ingest/manual/upload"
    headers = {"X-Admin-Key": admin_key} if admin_key else {}

    if file_path.suffix.lower() == ".docx":
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        filename = file_path.name
        content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    else:
        # .txt - convert to docx
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        file_bytes = _txt_to_docx_bytes(text)
        filename = file_path.stem + ".docx"
        content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    files = {"file": (filename, file_bytes, content_type)}
    data = {
        "company_ticker": ticker,
        "call_date": call_date,
        "use_eval_table": str(use_eval_table).lower(),
    }

    resp = httpx.post(url, headers=headers, files=files, data=data, timeout=120.0)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch ingest transcripts from transcripts/{ticker}/{date}.docx")
    parser.add_argument(
        "--dir",
        "-d",
        type=Path,
        default=DEFAULT_TRANSCRIPTS_DIR,
        help=f"Transcripts directory (default: {DEFAULT_TRANSCRIPTS_DIR})",
    )
    parser.add_argument(
        "--base-url",
        "-u",
        default=os.environ.get("BASE_URL", DEFAULT_BASE_URL),
        help=f"API base URL (default: from BASE_URL env or {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--admin-key",
        default=os.environ.get("ADMIN_KEY") or os.environ.get("ADMIN_API_KEY", ""),
        help="Admin API key (X-Admin-Key)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Ingest into eval_document_chunks instead of document_chunks",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be ingested without uploading",
    )
    args = parser.parse_args()

    transcripts_dir = args.dir.resolve()
    files = _discover_files(transcripts_dir)

    if not files:
        print(f"No transcript files found in {transcripts_dir}")
        print(f"Expected structure: {transcripts_dir}/{{TICKER}}/{{YYYY-MM-DD}}.docx")
        sys.exit(0)

    print(f"Found {len(files)} transcript(s) in {transcripts_dir}:")
    for f, ticker, call_date in files:
        print(f"  {ticker}/{f.name} -> ticker={ticker}, call_date={call_date}")

    if args.dry_run:
        print("\n[DRY RUN] No uploads performed.")
        return

    print(f"\nUploading to {args.base_url}...")
    success = 0
    for file_path, ticker, call_date in files:
        try:
            result = upload_file(
                file_path, ticker, call_date,
                base_url=args.base_url,
                admin_key=args.admin_key,
                use_eval_table=args.eval,
            )
            chunks = result.get("chunks_created", 0)
            print(f"  [OK] {ticker}/{file_path.name} -> {chunks} chunks (doc_id={result.get('doc_id', '')})")
            success += 1
        except httpx.HTTPStatusError as e:
            print(f"  [FAIL] {ticker}/{file_path.name}: {e.response.status_code} - {e.response.text[:150]}")
        except Exception as e:
            print(f"  [FAIL] {ticker}/{file_path.name}: {e}")

    print(f"\nDone. {success}/{len(files)} ingested successfully.")


if __name__ == "__main__":
    main()
