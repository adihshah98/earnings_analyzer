"""Eval harness runner using PydanticEvals."""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic_evals.reporting import EvaluationReport

from app.agents.streaming import run_simple_rag
from app.evals.datasets import load_dataset
from app.models.schemas import (
    AgentResponse,
    EvalCaseResult,
    EvalRunResult,
    MetricScore,
)

logger = logging.getLogger(__name__)


async def _task(
    inputs: dict[str, str],
    search_mode: str | None = None,
) -> AgentResponse:
    """Task function for PydanticEvals: runs query through simple RAG (same as prod)."""
    return await run_simple_rag(
        query=inputs["query"],
        search_mode=search_mode,
    )


def _report_to_eval_run_result(
    report: EvaluationReport,
    dataset_name: str,
    run_id: str,
    total_latency_ms: float,
) -> EvalRunResult:
    """Map PydanticEvals EvaluationReport to our EvalRunResult schema."""
    case_results: list[EvalCaseResult] = []
    metric_scores: dict[str, list[float]] = {}

    def _parse_scores(scores: dict[str, Any]) -> list[MetricScore]:
        metric_list: list[MetricScore] = []
        for name, val in scores.items():
            if hasattr(val, "value"):
                score_val = val.value
                details_val = getattr(val, "reason", "") or ""
            else:
                score_val = val
                details_val = ""
            metric_list.append(
                MetricScore(
                    name=name,
                    score=float(score_val) if isinstance(score_val, (int, float)) else 0.0,
                    details=str(details_val),
                )
            )
        for m in metric_list:
            metric_scores.setdefault(m.name, []).append(m.score)
        return metric_list

    for case in report.cases:
        query = case.inputs.get("query", "") if isinstance(case.inputs, dict) else str(case.inputs)
        output = case.output
        answer = output.answer if isinstance(output, AgentResponse) else str(output) if output else ""

        case_results.append(
            EvalCaseResult(
                query=query,
                response=answer,
                scores=_parse_scores(case.scores),
                latency_ms=case.task_duration * 1000,
            )
        )

    for failure in report.failures:
        query = failure.inputs.get("query", "") if isinstance(failure.inputs, dict) else str(failure.inputs)
        case_results.append(
            EvalCaseResult(
                query=query,
                response=f"ERROR: {failure.error_message}",
                scores=_parse_scores({}),
                latency_ms=0.0,
            )
        )

    overall_scores = {
        name: sum(vals) / len(vals) if vals else 0.0
        for name, vals in metric_scores.items()
    }

    return EvalRunResult(
        run_id=run_id,
        dataset_name=dataset_name,
        overall_scores=overall_scores,
        case_results=case_results,
        total_latency_ms=total_latency_ms,
        created_at=datetime.now(timezone.utc),
    )


async def run_eval(
    dataset_name: str,
    concurrency: int = 3,
    search_mode: str | None = None,
    tag_filter: list[str] | None = None,
) -> EvalRunResult:
    """Run a full evaluation suite using PydanticEvals.

    Executes all cases in an eval dataset against the agent via Dataset.evaluate(),
    using custom evaluators (faithfulness, relevance, completeness) and mapping
    the PydanticEvals report to our EvalRunResult schema for API compatibility.

    For retrieval quality metrics (precision@k, recall@k, MRR, hit@k),
    use scripts/run_retrieval_evals.py which calls the retriever directly.

    Args:
        dataset_name: Name of the eval dataset to run.
        concurrency: Max concurrent eval cases.
        search_mode: Retrieval mode (vector, keyword, hybrid). None defaults to hybrid.
        tag_filter: If set, only run cases whose metadata.tags contain at least one of these.

    Returns:
        EvalRunResult with aggregated scores and case details.
    """
    run_id = str(uuid.uuid4())[:8]
    dataset = load_dataset(dataset_name, tag_filter=tag_filter)
    if not dataset.cases:
        raise ValueError(
            f"No cases to run for dataset '{dataset_name}'"
            + (f" with tag_filter={tag_filter}" if tag_filter else "")
        )

    mode = search_mode or "hybrid"
    logger.info(
        f"Starting eval run {run_id}: dataset='{dataset_name}', "
        f"cases={len(dataset.cases)}, search_mode={mode}"
    )

    def make_task(sm: str | None):
        async def task(inputs: dict[str, str]) -> AgentResponse:
            return await _task(inputs, search_mode=sm)

        return task

    report = await dataset.evaluate(
        make_task(mode),
        name=f"simple_rag_{run_id}",
        max_concurrency=concurrency,
        progress=True,
    )

    total_latency_ms = sum(c.task_duration for c in report.cases) * 1000

    result = _report_to_eval_run_result(
        report,
        dataset_name=dataset_name,
        run_id=run_id,
        total_latency_ms=total_latency_ms,
    )

    logger.info(
        f"Eval run {run_id} complete: {result.overall_scores}, "
        f"total_latency={total_latency_ms:.0f}ms"
    )

    return result
