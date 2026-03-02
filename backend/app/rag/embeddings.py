"""OpenAI embedding generation utilities."""

import logging
from typing import Sequence

import numpy as np
from openai import AsyncOpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None


def get_openai_client() -> AsyncOpenAI:
    """Get or create the OpenAI async client."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


async def generate_embedding(text: str) -> list[float]:
    """Generate an embedding for a single text string."""
    settings = get_settings()
    client = get_openai_client()

    text = text.replace("\n", " ").strip()
    if not text:
        return [0.0] * settings.embedding_dimensions

    response = await client.embeddings.create(
        input=text,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )

    return response.data[0].embedding


async def generate_embeddings_batch(
    texts: Sequence[str],
    batch_size: int = 100,
) -> list[list[float]]:
    """Generate embeddings for a batch of texts.

    Processes in batches to respect API limits.
    """
    settings = get_settings()
    client = get_openai_client()
    all_embeddings: list[list[float]] = []

    cleaned = [t.replace("\n", " ").strip() for t in texts]

    for i in range(0, len(cleaned), batch_size):
        batch = cleaned[i : i + batch_size]
        batch = [t if t else "empty" for t in batch]

        response = await client.embeddings.create(
            input=batch,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )

        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

        logger.info(f"Embedded batch {i // batch_size + 1}, total: {len(all_embeddings)}")

    return all_embeddings


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))
