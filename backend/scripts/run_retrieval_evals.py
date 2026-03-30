"""Run retrieval quality evals.

Usage:
  cd backend && python scripts/run_retrieval_evals.py [dataset]
  cd backend && python scripts/run_retrieval_evals.py [dataset] --remote --base-url http://localhost:8000

Default: in-process against your DB (eval chunks), with live per-case progress like run_evals.py.
Use --remote to call POST /evals/retrieval instead (single request; no per-case lines here).

Remote defaults: base-url from BASE_URL env or deployment URL, admin-key from ADMIN_KEY or ADMIN_API_KEY.
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent
DEFAULT_EVAL_RESULTS_DIR = BACKEND_ROOT / "eval_results"
DEFAULT_BASE_URL = "https://earnings-analyzer-dud6.onrender.com"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Load backend/.env so BASE_URL and ADMIN_KEY/ADMIN_API_KEY are available
if load_dotenv:
    load_dotenv(BACKEND_ROOT / ".env")


async def _list_datasets(base_url: str, admin_key: str) -> list[str]:
    """Call GET /evals/datasets."""
    try:
        import httpx
    except ImportError:
        print("ERROR: httpx required for HTTP mode. pip install httpx")
        sys.exit(1)

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
) -> dict:
    """Call POST /evals/retrieval."""
    try:
        import httpx
    except ImportError:
        print("ERROR: httpx required for HTTP mode. pip install httpx")
        sys.exit(1)

    url = f"{base_url.rstrip('/')}/evals/retrieval"
    params = {"dataset_name": dataset_name, "progress": True}
    headers = {"X-Admin-Key": admin_key} if admin_key else {}
    async with httpx.AsyncClient(timeout=900.0) as client:
        resp = await client.post(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()


async def _run_retrieval_eval_local(dataset_name: str) -> dict:
    """Run retrieval eval in-process (same terminal as stderr progress lines)."""
    from app.evals.context import use_eval_chunks_context
    from app.evals.retrieval import retrieval_eval_to_dict, run_retrieval_eval

    async with use_eval_chunks_context():
        result = await run_retrieval_eval(
            dataset_name=dataset_name,
            progress=True,
        )
    return retrieval_eval_to_dict(result)


def _print_results(result: dict) -> None:
    scores = result.get("scores_by_mode", {})
    print(f"\nRun ID: {result.get('run_id', 'N/A')}")
    print(f"Dataset: {result.get('dataset_name')}")
    print(f"Cases: {result.get('n_positive_cases', '?')} positive, {result.get('n_negative_cases', '?')} negative (excluded from scores)")
    print("\nOverall scores by mode:")
    for mode, s in scores.items():
        if isinstance(s, dict):
            n = sum(r.get("results_by_mode", {}).get(mode, {}).get("num_returned", 0)
                    for r in result.get("case_details", []))
            cases = len(result.get("case_details", []))
            avg_n = n / cases if cases else 0
            print(f"  {mode} (avg {avg_n:.1f} chunks/case):")
            print(f"    precision: {s.get('precision', 0):.3f}")
            print(f"    recall:    {s.get('recall', 0):.3f}")
            print(f"    MRR:       {s.get('mrr', 0):.3f}")
            print(f"    hit:       {s.get('hit', 0):.3f}")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run retrieval evals (default: in-process with live progress; use --remote for HTTP)",
    )
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
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Call POST /evals/retrieval over HTTP instead of in-process",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_EVAL_RESULTS_DIR

    if args.list:
        if args.remote:
            datasets = await _list_datasets(args.base_url, args.admin_key)
        else:
            from app.evals.datasets import list_datasets

            datasets = list_datasets()
        print("Available datasets:")
        for ds in datasets:
            print(f"  - {ds}")
        return

    if args.remote:
        datasets_to_run = (
            [args.dataset]
            if args.dataset
            else await _list_datasets(args.base_url, args.admin_key)
        )
    else:
        from app.evals.datasets import list_datasets

        datasets_to_run = (
            [args.dataset]
            if args.dataset
            else list_datasets()
        )
    if not datasets_to_run:
        print("No datasets available. Ensure the API is running and has eval_datasets/.")
        sys.exit(1)
    if not args.dataset and len(datasets_to_run) > 1:
        print(f"Running all {len(datasets_to_run)} datasets: {datasets_to_run}")

    for ds_name in datasets_to_run:
        target = args.base_url if args.remote else "(in-process)"
        print(f"\nRunning eval: dataset='{ds_name}' → {target}")
        print("-" * 60)
        try:
            if args.remote:
                result = await _run_retrieval_eval(args.base_url, args.admin_key, ds_name)
            else:
                result = await _run_retrieval_eval_local(ds_name)
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
        except Exception as e:
            resp = getattr(e, "response", None)
            if resp is not None and hasattr(resp, "status_code"):
                text = getattr(resp, "text", "")[:200]
                print(f"ERROR: {resp.status_code} - {text}")
            else:
                print(f"ERROR: {e}")
            sys.exit(1)

    if len(datasets_to_run) > 1:
        print(f"\n{'=' * 60}\nCompleted {len(datasets_to_run)} dataset(s)")


if __name__ == "__main__":
    asyncio.run(main())
