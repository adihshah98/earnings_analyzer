"""Run agent answer quality evals by calling the deployed API.

Usage:
  cd backend && python scripts/run_evals.py [dataset] --base-url http://localhost:8000
  BASE_URL=https://api.example.com ADMIN_KEY=secret python scripts/run_evals.py acme_eval

Defaults: base-url=http://localhost:8000, admin-key from ADMIN_KEY env.
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

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent
DEFAULT_EVAL_RESULTS_DIR = BACKEND_ROOT / "eval_results"
DEFAULT_BASE_URL = "http://localhost:8000"


async def _list_datasets(base_url: str, admin_key: str) -> list[str]:
    """Call GET /evals/datasets."""
    url = f"{base_url.rstrip('/')}/evals/datasets"
    headers = {"X-Admin-Key": admin_key} if admin_key else {}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json()


async def _run_agent_eval(
    base_url: str,
    admin_key: str,
    dataset_name: str,
) -> dict:
    """Call POST /evals/run."""
    url = f"{base_url.rstrip('/')}/evals/run"
    params = {"dataset_name": dataset_name}
    headers = {"X-Admin-Key": admin_key} if admin_key else {}
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()


def _print_results(result: dict) -> None:
    print(f"\nRun ID: {result['run_id']}")
    print(f"Dataset: {result['dataset_name']}")
    print(f"Cases: {len(result['case_results'])}")
    print(f"Total latency: {result['total_latency_ms']:.0f}ms")
    print("Overall Scores:")
    for metric, score in result.get("overall_scores", {}).items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {metric:25s} {bar} {score:.3f}")
    print("\nPer-case Results (sample):")
    for cr in result.get("case_results", [])[:5]:
        scores_str = ", ".join(f"{s['name']}={s['score']:.2f}" for s in cr.get("scores", []))
        print(f"  Q: {cr['query'][:60]}...")
        print(f"     {scores_str} ({cr.get('latency_ms', 0):.0f}ms)")
    if len(result.get("case_results", [])) > 5:
        print(f"  ... and {len(result['case_results']) - 5} more")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent evals via deployed API")
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
        default=os.environ.get("ADMIN_KEY", ""),
        help="Admin API key (X-Admin-Key). Use ADMIN_KEY env for local dev with empty key.",
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
        print(f"\nRunning eval: dataset='{ds_name}' → {args.base_url}")
        print("-" * 60)
        try:
            result = await _run_agent_eval(args.base_url, args.admin_key, ds_name)
            _print_results(result)

            output_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_path = (
                Path(args.output)
                if args.output and len(datasets_to_run) == 1
                else output_dir / f"{ds_name}_{result['run_id']}_{ts}.json"
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
