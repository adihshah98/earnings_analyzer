"""Evaluation metrics for agent quality assessment."""

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from app.config import get_settings
from app.prompts.templates import FAITHFULNESS_EVAL_PROMPT, RELEVANCE_EVAL_PROMPT
from app.rag.embeddings import cosine_similarity, generate_embedding

logger = logging.getLogger(__name__)


async def _llm_judge(prompt: str) -> dict[str, Any]:
    """Run an LLM-as-judge evaluation.

    Uses OpenAI Responses API (not chat.completions).

    Returns:
        Dict with 'score' and 'reason' keys.
    """
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    response = await client.responses.create(
        model=settings.openai_model,
        input=prompt,
    )

    # Use output_text property (Responses API); fallback for older SDK
    raw = (
        response.output_text
        if hasattr(response, "output_text")
        else _extract_text_from_responses_output(response.output)
    ).strip()

    # Parse JSON response
    try:
        # Handle potential markdown code blocks
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
        return result
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse LLM judge response: {raw}")
        return {"score": 0.0, "reason": "Failed to parse judge response"}


def _extract_text_from_responses_output(output: list) -> str:
    """Extract text from Responses API output (fallback when output_text unavailable)."""
    parts = []
    for item in output or []:
        if getattr(item, "type", None) == "message" and hasattr(item, "content"):
            for c in item.content or []:
                if getattr(c, "type", None) == "output_text":
                    parts.append(getattr(c, "text", "") or "")
    return "".join(parts)


async def faithfulness_score(
    question: str,
    context: str,
    answer: str,
) -> dict[str, Any]:
    """Score how faithful the answer is to the provided context.

    Uses LLM-as-judge to evaluate whether the answer is grounded
    in the context documents.

    Args:
        question: The original question.
        context: The context documents provided to the agent.
        answer: The agent's answer.

    Returns:
        Dict with 'score' (0-1) and 'reason'.
    """
    prompt = FAITHFULNESS_EVAL_PROMPT.format(
        question=question,
        context=context,
        answer=answer,
    )

    result = await _llm_judge(prompt)
    return {
        "name": "faithfulness",
        "score": float(result.get("score", 0.0)),
        "details": result.get("reason", ""),
    }


async def relevance_score(
    question: str,
    answer: str,
) -> dict[str, Any]:
    """Score how relevant the answer is to the question.

    Uses LLM-as-judge to evaluate answer relevance.

    Args:
        question: The original question.
        answer: The agent's answer.

    Returns:
        Dict with 'score' (0-1) and 'reason'.
    """
    prompt = RELEVANCE_EVAL_PROMPT.format(
        question=question,
        answer=answer,
    )

    result = await _llm_judge(prompt)
    return {
        "name": "relevance",
        "score": float(result.get("score", 0.0)),
        "details": result.get("reason", ""),
    }


async def semantic_similarity_score(
    expected: str,
    actual: str,
) -> dict[str, Any]:
    """Score semantic similarity between expected and actual answers.

    Uses embedding cosine similarity as a metric.

    Args:
        expected: The expected/reference answer.
        actual: The actual agent answer.

    Returns:
        Dict with 'score' (0-1) and 'details'.
    """
    expected_emb = await generate_embedding(expected)
    actual_emb = await generate_embedding(actual)

    similarity = cosine_similarity(expected_emb, actual_emb)

    return {
        "name": "semantic_similarity",
        "score": max(0.0, min(1.0, similarity)),
        "details": f"Cosine similarity: {similarity:.4f}",
    }


async def answer_completeness_score(
    question: str,
    expected: str | None,
    actual: str,
) -> dict[str, Any]:
    """Score how complete the answer is compared to expectations.

    Uses LLM-as-judge to check whether all key facts from the expected
    answer appear in the actual answer. Does not penalize extra detail.
    Falls back to a general completeness check if no expected answer.

    Args:
        question: The original question.
        expected: Optional expected answer for comparison.
        actual: The actual agent answer.

    Returns:
        Dict with 'score' (0-1) and 'details'.
    """
    if expected:
        from app.prompts.templates import COMPLETENESS_EVAL_PROMPT

        prompt = COMPLETENESS_EVAL_PROMPT.format(
            question=question,
            expected=expected,
            answer=actual,
        )
        result = await _llm_judge(prompt)
        return {
            "name": "completeness",
            "score": float(result.get("score", 0.0)),
            "details": result.get("reason", ""),
        }

    prompt = f"""Rate the completeness of this answer on a 0.0-1.0 scale.

Question: {question}
Answer: {actual}

Respond with ONLY a JSON object: {{"score": <float>, "reason": "<brief explanation>"}}"""

    result = await _llm_judge(prompt)
    return {
        "name": "completeness",
        "score": float(result.get("score", 0.0)),
        "details": result.get("reason", ""),
    }
