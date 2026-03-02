"""PydanticEvals evaluator classes wrapping the metric functions in metrics.py."""

from dataclasses import dataclass

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from app.evals.metrics import answer_completeness_score, faithfulness_score, relevance_score
from app.models.schemas import AgentResponse


def _extract_context(response: AgentResponse) -> str:
    """Build a context string from the source documents attached to a response.

    Prefers raw retrieved chunks (the actual retriever output) over agent-cited
    sources, since the agent may not populate sources in its structured output.
    """
    raw_chunks = getattr(response, "_raw_retrieved_chunks", None) or []
    if raw_chunks:
        return "\n\n".join(
            f"[{c.get('metadata', {}).get('title', c.get('chunk_id', 'Unknown'))}]\n{c.get('content', '')}"
            for c in raw_chunks
        )
    if response.sources:
        return "\n\n".join(
            f"[{s.metadata.get('title', s.chunk_id)}]\n{s.content}" for s in response.sources
        )
    return ""


@dataclass
class FaithfulnessEvaluator(Evaluator):
    async def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        output: AgentResponse = ctx.output
        query = ctx.inputs.get("query", "") if isinstance(ctx.inputs, dict) else str(ctx.inputs)
        context = _extract_context(output)
        if not context:
            return EvaluationReason(value=0.0, reason="No retrieval context available to verify against")
        result = await faithfulness_score(query, context, output.answer)
        return EvaluationReason(
            value=float(result["score"]),
            reason=result.get("details", ""),
        )


@dataclass
class RelevanceEvaluator(Evaluator):
    async def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        output: AgentResponse = ctx.output
        query = ctx.inputs.get("query", "") if isinstance(ctx.inputs, dict) else str(ctx.inputs)
        result = await relevance_score(query, output.answer)
        return EvaluationReason(
            value=float(result["score"]),
            reason=result.get("details", ""),
        )


@dataclass
class CompletenessEvaluator(Evaluator):
    async def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        output: AgentResponse = ctx.output
        query = ctx.inputs.get("query", "") if isinstance(ctx.inputs, dict) else str(ctx.inputs)
        expected = ctx.expected_output if isinstance(ctx.expected_output, str) else None
        result = await answer_completeness_score(query, expected, output.answer)
        return EvaluationReason(
            value=float(result["score"]),
            reason=result.get("details", ""),
        )
