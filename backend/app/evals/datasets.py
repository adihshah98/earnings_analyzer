"""Eval dataset loading and management with PydanticEvals."""

import json
import logging
from pathlib import Path
from typing import Any

from pydantic_evals import Case, Dataset

from app.evals.evaluators import CompletenessEvaluator, FaithfulnessEvaluator, RelevanceEvaluator
from app.models.schemas import AgentResponse

logger = logging.getLogger(__name__)

# PydanticEvals types: inputs=query dict, output=AgentResponse, metadata=dict
EvalInputsT = dict[str, str]
EvalOutputT = AgentResponse
EvalMetadataT = dict[str, Any]

# Dataset directory: backend/eval_datasets
_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DATASETS_DIR = _BACKEND_ROOT / "eval_datasets"

# Dataset-level evaluators for all KB agent evals
DEFAULT_EVALUATORS = [
    FaithfulnessEvaluator(),
    RelevanceEvaluator(),
    CompletenessEvaluator(),
]


def load_dataset(
    name: str,
    tag_filter: list[str] | None = None,
) -> Dataset[EvalInputsT, str | None, EvalMetadataT]:
    """Load an eval dataset by name and convert to PydanticEvals Dataset.

    Loads from our JSON format (query, expected_answer, expected_sources, tags)
    and converts to PydanticEvals Case/Dataset with custom evaluators.

    Args:
        name: Dataset name (without .json extension).
        tag_filter: If set, only include cases whose metadata.tags contain
            at least one of these tags.

    Returns:
        PydanticEvals Dataset ready for evaluate().
    """
    path = EVAL_DATASETS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Eval dataset not found: {path}")

    with open(path) as f:
        data = json.load(f)

    cases: list[Case[EvalInputsT, str | None, EvalMetadataT]] = []
    for i, raw in enumerate(data.get("cases", [])):
        query = raw.get("query", "")
        expected_answer = raw.get("expected_answer")
        expected_sources = raw.get("expected_sources", [])
        tags = raw.get("tags", [])

        if tag_filter and not any(t in tag_filter for t in tags):
            continue

        case_name = raw.get("name") or f"Case {i + 1}"
        case = Case(
            name=case_name,
            inputs={"query": query},
            expected_output=expected_answer if expected_answer else None,
            metadata={
                "expected_sources": expected_sources,
                "tags": tags,
            },
        )
        cases.append(case)

    return Dataset(
        name=data.get("name", name),
        cases=cases,
        evaluators=DEFAULT_EVALUATORS,
    )


def list_datasets() -> list[str]:
    """List available eval datasets."""
    if not EVAL_DATASETS_DIR.exists():
        return []

    return [
        p.stem
        for p in EVAL_DATASETS_DIR.glob("*.json")
    ]
