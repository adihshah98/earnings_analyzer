"""Run retrieval quality evals by calling the deployed API.

Usage:
  cd backend && python scripts/run_retrieval_evals.py [dataset]
  cd backend && python scripts/run_retrieval_evals.py [dataset] --base-url http://localhost:8000

Defaults: base-url from BASE_URL env or deployment URL, admin-key from ADMIN_KEY or ADMIN_API_KEY env.
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
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

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent
DEFAULT_EVAL_RESULTS_DIR = BACKEND_ROOT / "eval_results"
DEFAULT_BASE_URL = "https://earnings-analyzer-dud6.onrender.com"

# Load backend/.env so BASE_URL and ADMIN_KEY/ADMIN_API_KEY are available
if load_dotenv:
    load_dotenv(BACKEND_ROOT / ".env")


async def _list_datasets(base_url: str, admin_key: str) -> list[str]:
    """Call GET /evals/datasets."""
    url = f"{base_url.rstrip('/')}/evals/datasets"
    headers = {"X-Admin-Key": admin_key} if admin_key else {}
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json()


async def _run_retrieval_eval(
    base_url: str,
    admin_key: str,
    dataset_name: str,
    top_k: int = 5,
) -> dict:
    """Call POST /evals/retrieval."""
    url = f"{base_url.rstrip('/')}/evals/retrieval"
    params = {"dataset_name": dataset_name, "top_k": top_k}
    headers = {"X-Admin-Key": admin_key} if admin_key else {}
    async with httpx.AsyncClient(timeout=900.0) as client:
        resp = await client.post(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()


def _print_results(result: dict) -> None:
    scores = result.get("scores_by_mode", {})
    print(f"\nRun ID: {result.get('run_id', 'N/A')}")
    print(f"Dataset: {result.get('dataset_name')}")
    print(f"Top-K: {result.get('top_k', 5)}")
    print(f"Cases: {len(result.get('case_details', []))}")
    print("\nOverall scores by mode:")
    for mode, s in scores.items():
        if isinstance(s, dict):
            print(f"  {mode}:")
            print(f"    precision@k: {s.get('precision_at_k', 0):.3f}")
            print(f"    recall@k:    {s.get('recall_at_k', 0):.3f}")
            print(f"    MRR:        {s.get('mrr', 0):.3f}")
            print(f"    hit@k:      {s.get('hit_at_k', 0):.3f}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval evals via deployed API")
    parser.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Dataset name, or omit to run all datasets",
    )
    parser.add_argument(
        "--base-url",
        "-u",
        default=os.environ.get("BASE_URL", DEFAULT_BASE_URL),
        help=f"API base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--admin-key",
        default=os.environ.get("ADMIN_KEY") or os.environ.get("ADMIN_API_KEY", ""),
        help="Admin API key (X-Admin-Key). From ADMIN_KEY or ADMIN_API_KEY env.",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per query (default: 5)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available datasets",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for JSON output (default: backend/eval_results)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_EVAL_RESULTS_DIR

    if args.list:
        datasets = await _list_datasets(args.base_url, args.admin_key)
        print("Available datasets:")
        for ds in datasets:
            print(f"  - {ds}")
        return

    datasets_to_run = (
        [args.dataset]
        if args.dataset
        else await _list_datasets(args.base_url, args.admin_key)
    )
    if not datasets_to_run:
        print("No datasets available. Ensure the API is running and has eval_datasets/.")
        sys.exit(1)
    if not args.dataset and len(datasets_to_run) > 1:
        print(f"Running all {len(datasets_to_run)} datasets: {datasets_to_run}")

    for ds_name in datasets_to_run:
        print(f"\nRunning retrieval eval: dataset='{ds_name}', top_k={args.top_k} → {args.base_url}")
        print("-" * 60)
        try:
            result = await _run_retrieval_eval(
                args.base_url, args.admin_key, ds_name, args.top_k
            )
            _print_results(result)

            output_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            run_id = result.get("run_id", "unknown")
            out_path = (
                Path(args.output)
                if args.output and len(datasets_to_run) == 1
                else output_dir / f"retrieval_{ds_name}_{run_id}_{ts}.json"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\nResults saved to {out_path}")
        except httpx.HTTPStatusError as e:
            print(f"ERROR: {e.response.status_code} - {e.response.text[:200]}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: {e}")
            sys.exit(1)

    if len(datasets_to_run) > 1:
        print(f"\n{'=' * 60}\nCompleted {len(datasets_to_run)} dataset(s)")


if __name__ == "__main__":
    asyncio.run(main())
