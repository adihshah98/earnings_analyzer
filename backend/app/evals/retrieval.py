"""Retrieval quality evals comparing vector, keyword, and hybrid search modes."""

import hashlib
import logging
import uuid

from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Any

from app.evals.datasets import load_dataset
from app.evals.metrics import _llm_judge
from app.rag.retriever import retrieve_relevant_chunks

logger = logging.getLogger(__name__)

# Cache LLM relevance judgments to avoid duplicate calls across modes.
# Key: hash(query + chunk_content) -> bool
_relevance_cache: dict[str, bool] = {}


@dataclass
class RetrievalModeScores:
    """Per-mode retrieval metrics."""

    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    hit_at_k: float = 0.0


@dataclass
class RetrievalEvalResult:
    """Result of running retrieval evals across modes."""

    run_id: str
    dataset_name: str
    top_k: int
    scores_by_mode: dict[str, RetrievalModeScores] = field(default_factory=dict)
    scores_by_tag: dict[str, dict[str, RetrievalModeScores]] = field(default_factory=dict)
    case_details: list[dict[str, Any]] = field(default_factory=list)


def _cache_key(query: str, content: str) -> str:
    return hashlib.sha256(f"{query}|||{content}".encode()).hexdigest()


def _title_matches(chunk: dict[str, Any], expected_sources: list[str]) -> bool:
    """Check if a chunk's title matches one of the expected sources."""
    title = chunk.get("metadata", {}).get("title")
    return title in expected_sources if title and expected_sources else False


async def _is_chunk_relevant(
    query: str,
    chunk: dict[str, Any],
    expected_sources: list[str],
) -> bool:
    """Check chunk relevance using both title matching and LLM-judged content relevance.

    A chunk is relevant if its title matches an expected source AND the LLM
    judge confirms its content helps answer the query.
    """
    if not _title_matches(chunk, expected_sources):
        return False

    content = chunk.get("content", "").strip()
    if not content:
        return False

    key = _cache_key(query, content)
    if key in _relevance_cache:
        return _relevance_cache[key]

    prompt = (
        "You are an IR relevance judge. Given a user query and a retrieved "
        "document chunk, decide whether the chunk contains information that "
        "helps answer the query.\n\n"
        f"Query: {query}\n\n"
        f"Chunk:\n{content}\n\n"
        'Respond with ONLY a JSON object: {"relevant": true, "reason": "..."} '
        'or {"relevant": false, "reason": "..."}'
    )

    try:
        result = await _llm_judge(prompt)
        relevant = bool(result.get("relevant", False))
    except Exception as e:
        logger.warning(f"LLM relevance judge failed, falling back to false: {e}")
        relevant = False

    _relevance_cache[key] = relevant
    return relevant


async def _compute_metrics(
    query: str,
    chunks: list[dict[str, Any]],
    expected_sources: list[str],
    top_k: int,
) -> tuple[float, float, float, float]:
    """Compute precision@k, recall@k, MRR, hit@k for a single query.

    precision@k and MRR use LLM-judged content relevance.
    recall@k uses expected_sources (did we find the right documents?).
    """
    top_chunks = chunks[:top_k]

    relevance_flags = [await _is_chunk_relevant(query, c, expected_sources) for c in top_chunks]
    num_relevant = sum(relevance_flags)

    precision = num_relevant / top_k if top_k else 0.0

    # recall: how many expected source documents appear in the results
    num_expected = len(expected_sources) if expected_sources else 1
    sources_found = {
        c.get("metadata", {}).get("title")
        for c, is_rel in zip(top_chunks, relevance_flags)
        if is_rel and c.get("metadata", {}).get("title")
    } & set(expected_sources)
    recall = len(sources_found) / num_expected if num_expected else 0.0

    mrr = 0.0
    for i, is_rel in enumerate(relevance_flags):
        if is_rel:
            mrr = 1.0 / (i + 1)
            break

    hit = 1.0 if num_relevant > 0 else 0.0

    return precision, recall, mrr, hit


def _chunk_to_serializable(chunk: dict[str, Any], relevant: bool | None = None) -> dict[str, Any]:
    """Reduce chunk to serializable form for debugging."""
    out: dict[str, Any] = {
        "content": chunk.get("content"),
        "metadata": chunk.get("metadata"),
        "similarity": chunk.get("similarity"),
    }
    if relevant is not None:
        out["llm_relevant"] = relevant
    return out


async def run_retrieval_eval(
    dataset_name: str,
    top_k: int = 5,
    save_chunks: bool = False,
    tag_filter: list[str] | None = None,
    progress: bool = True,
) -> RetrievalEvalResult:
    """Run retrieval evals across vector, keyword, and hybrid modes.

    Calls retrieve_relevant_chunks directly (bypasses the LLM agent) so
    metrics reflect actual retriever quality, not agent citation behavior.
    Uses LLM-as-judge for chunk relevance and expected_sources for recall.

    Args:
        dataset_name: Eval dataset name (e.g. acme_eval).
        top_k: Number of results to retrieve per query.
        save_chunks: If True, include retrieved chunks in case_details for debugging.
        tag_filter: If set, only run cases whose tags contain at least one match.
        progress: If True, show a tqdm progress bar over cases (no per-query text).

    Returns:
        RetrievalEvalResult with scores by mode and per-case details.
    """
    _relevance_cache.clear()

    run_id = str(uuid.uuid4())[:8]
    dataset = load_dataset(dataset_name, tag_filter=tag_filter)

    modes: list[str] = ["vector", "keyword", "hybrid"]
    mode_scores: dict[str, list[tuple[float, float, float, float]]] = {
        m: [] for m in modes
    }
    tag_mode_scores: dict[str, dict[str, list[tuple[float, float, float, float]]]] = {}
    case_details: list[dict[str, Any]] = []

    cases_to_run = []
    for case in dataset.cases:
        q = case.inputs.get("query", "") if isinstance(case.inputs, dict) else str(case.inputs)
        if q:
            cases_to_run.append(case)

    logger.info(
        f"Starting retrieval eval {run_id}: dataset='{dataset_name}', "
        f"cases={len(cases_to_run)}, top_k={top_k}"
    )

    n_cases = len(cases_to_run)
    case_iter = enumerate(cases_to_run, start=1)
    if progress and n_cases > 0:
        case_iter = tqdm(
            case_iter,
            total=n_cases,
            desc=f"retrieval {run_id}",
            unit="case",
            leave=True,
        )
    for _, case in case_iter:
        query = case.inputs.get("query", "") if isinstance(case.inputs, dict) else str(case.inputs)
        case_meta = case.metadata if isinstance(case.metadata, dict) else {}
        expected_sources: list[str] = case_meta.get("expected_sources", [])
        tags: list[str] = case_meta.get("tags", [])

        case_result: dict[str, Any] = {
            "query": query,
            "expected_sources": expected_sources,
            "tags": tags,
            "results_by_mode": {},
        }

        for mode in modes:
            chunks = await retrieve_relevant_chunks(
                query=query,
                top_k=top_k,
                search_mode=mode,
            )
            prec, rec, mrr, hit = await _compute_metrics(
                query, chunks, expected_sources, top_k
            )
            mode_scores[mode].append((prec, rec, mrr, hit))
            for tag in tags:
                if tag not in tag_mode_scores:
                    tag_mode_scores[tag] = {m: [] for m in modes}
                tag_mode_scores[tag][mode].append((prec, rec, mrr, hit))
            mode_result: dict[str, Any] = {
                "num_returned": len(chunks),
                "precision_at_k": prec,
                "recall_at_k": rec,
                "mrr": mrr,
                "hit_at_k": hit,
            }
            if save_chunks:
                top_chunks = chunks[:top_k]
                relevance_flags = [await _is_chunk_relevant(query, c, expected_sources) for c in top_chunks]
                mode_result["chunks"] = [
                    _chunk_to_serializable(c, rel)
                    for c, rel in zip(top_chunks, relevance_flags)
                ]
            case_result["results_by_mode"][mode] = mode_result

        case_details.append(case_result)

    def _aggregate_scores(
        lst: list[tuple[float, float, float, float]],
    ) -> RetrievalModeScores:
        if not lst:
            return RetrievalModeScores()
        n = len(lst)
        return RetrievalModeScores(
            precision_at_k=sum(x[0] for x in lst) / n,
            recall_at_k=sum(x[1] for x in lst) / n,
            mrr=sum(x[2] for x in lst) / n,
            hit_at_k=sum(x[3] for x in lst) / n,
        )

    scores_by_mode: dict[str, RetrievalModeScores] = {
        mode: _aggregate_scores(mode_scores[mode]) for mode in modes
    }

    scores_by_tag: dict[str, dict[str, RetrievalModeScores]] = {}
    for tag, tag_modes in tag_mode_scores.items():
        scores_by_tag[tag] = {
            mode: _aggregate_scores(tag_modes[mode]) for mode in modes
        }

    result = RetrievalEvalResult(
        run_id=run_id,
        dataset_name=dataset_name,
        top_k=top_k,
        scores_by_mode=scores_by_mode,
        scores_by_tag=scores_by_tag,
        case_details=case_details,
    )

    logger.info(
        f"Retrieval eval {run_id} complete: "
        f"vector={scores_by_mode.get('vector')}, "
        f"keyword={scores_by_mode.get('keyword')}, "
        f"hybrid={scores_by_mode.get('hybrid')}"
    )
    return result


def _mode_scores_to_dict(s: RetrievalModeScores) -> dict[str, float]:
    return {
        "precision_at_k": s.precision_at_k,
        "recall_at_k": s.recall_at_k,
        "mrr": s.mrr,
        "hit_at_k": s.hit_at_k,
    }


def retrieval_eval_to_dict(result: RetrievalEvalResult) -> dict[str, Any]:
    """Convert RetrievalEvalResult to API-serializable dict."""
    out: dict[str, Any] = {
        "run_id": result.run_id,
        "dataset_name": result.dataset_name,
        "top_k": result.top_k,
        "scores_by_mode": {
            mode: _mode_scores_to_dict(s)
            for mode, s in result.scores_by_mode.items()
        },
        "case_details": result.case_details,
    }
    if result.scores_by_tag:
        out["scores_by_tag"] = {
            tag: {
                mode: _mode_scores_to_dict(s)
                for mode, s in tag_modes.items()
            }
            for tag, tag_modes in result.scores_by_tag.items()
        }
    return out
