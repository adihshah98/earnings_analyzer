"""Retrieval quality evals comparing vector, keyword, and hybrid search modes."""

import hashlib
import logging
import uuid

from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Any

from app.agents.streaming import build_retrieval_plan, retrieve_from_plan
from app.evals.datasets import load_dataset
from app.evals.metrics import _llm_judge
from app.rag.retriever import get_available_periods, get_known_companies

logger = logging.getLogger(__name__)

# Cache LLM relevance judgments to avoid duplicate calls across modes.
# Key: hash(query + chunk_content) -> bool
_relevance_cache: dict[str, bool] = {}


@dataclass
class RetrievalModeScores:
    """Per-mode retrieval metrics — all evaluated over the full returned set."""

    precision: float = 0.0   # relevant / n_returned
    recall: float = 0.0      # sources_found / expected_sources
    mrr: float = 0.0         # 1 / rank_of_first_relevant
    hit: float = 0.0         # 1 if any relevant chunk returned


@dataclass
class RetrievalEvalResult:
    """Result of running retrieval evals across modes."""

    run_id: str
    dataset_name: str
    scores_by_mode: dict[str, RetrievalModeScores] = field(default_factory=dict)
    scores_by_tag: dict[str, dict[str, RetrievalModeScores]] = field(default_factory=dict)
    case_details: list[dict[str, Any]] = field(default_factory=list)


def _cache_key(query: str, content: str) -> str:
    return hashlib.sha256(f"{query}|||{content}".encode()).hexdigest()


def _source_matches(chunk: dict[str, Any], expected_sources: list[str]) -> bool:
    """Check if a chunk comes from one of the expected source documents.

    Matches on metadata.title first; falls back to reconstructing the title
    from (company_ticker, call_date) so minor title format variations don't
    cause false misses.
    """
    if not expected_sources:
        return False
    meta = chunk.get("metadata", {})
    title = meta.get("title")
    if title and title in expected_sources:
        return True
    # Fallback: reconstruct canonical title from ticker + call_date
    ticker = meta.get("company_ticker", "")
    call_date = meta.get("call_date", "")
    if ticker and call_date:
        return f"{ticker} Earnings Call {call_date}" in expected_sources
    return False


async def _is_chunk_relevant(
    query: str,
    chunk: dict[str, Any],
    expected_sources: list[str],
) -> bool:
    """Check chunk relevance: must come from an expected source document AND
    be judged helpful by the LLM.
    """
    if not _source_matches(chunk, expected_sources):
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
) -> tuple[float, float, float, float]:
    """Compute precision, recall, MRR, hit over ALL returned chunks.

    The production pipeline returns a variable-size set (8 for single-pair,
    3×n_pairs + n_pairs financials for multi-pair), so there is no meaningful
    fixed k. All four metrics are evaluated over the full returned list.

    - precision  = relevant_chunks / n_returned
    - recall     = sources_found / expected_sources
    - MRR        = 1 / rank_of_first_relevant
    - hit        = 1 if any relevant chunk is returned
    """
    if not chunks:
        return 0.0, 0.0, 0.0, 0.0

    # Run the cheap source-match gate first; only call the LLM judge when
    # the chunk comes from an expected source document.
    flags = []
    for c in chunks:
        if _source_matches(c, expected_sources):
            flags.append(await _is_chunk_relevant(query, c, expected_sources))
        else:
            flags.append(False)

    num_relevant = sum(flags)
    n = len(chunks)

    precision = num_relevant / n

    num_expected = len(expected_sources) if expected_sources else 1
    sources_found = {
        c.get("metadata", {}).get("title") or (
            f"{c.get('metadata', {}).get('company_ticker', '')} Earnings Call "
            f"{c.get('metadata', {}).get('call_date', '')}"
        )
        for c, is_rel in zip(chunks, flags)
        if is_rel
    } & set(expected_sources)
    recall = len(sources_found) / num_expected

    mrr = 0.0
    for i, is_rel in enumerate(flags):
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
    save_chunks: bool = False,
    tag_filter: list[str] | None = None,
    progress: bool = True,
) -> RetrievalEvalResult:
    """Run retrieval evals across vector, keyword, and hybrid modes.

    Mirrors the full production retrieval pipeline (entity resolution, query
    rewriting, filter_metadata, RRF merge, per-pair selection, financials
    injection) and evaluates precision, recall, MRR, and hit over the complete
    returned set — no fixed top_k cutoff.

    Args:
        dataset_name: Eval dataset name (e.g. prod_tickers_eval).
        save_chunks: If True, include retrieved chunks in case_details for debugging.
        tag_filter: If set, only run cases whose tags contain at least one of these.
        progress: If True, show a tqdm progress bar over cases.

    Returns:
        RetrievalEvalResult with scores by mode and per-case details.
    """
    _relevance_cache.clear()

    run_id = str(uuid.uuid4())[:8]
    dataset = load_dataset(dataset_name, tag_filter=tag_filter)

    # Pre-fetch companies and available periods once for the whole run;
    # both respect the eval chunks context so they return eval-table data.
    companies = await get_known_companies()
    available_periods = await get_available_periods()

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
        f"Starting retrieval eval {run_id}: dataset='{dataset_name}', cases={len(cases_to_run)}"
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
        conversation_context: list[str] | None = (
            case.inputs.get("conversation_context") if isinstance(case.inputs, dict) else None
        )
        case_meta = case.metadata if isinstance(case.metadata, dict) else {}
        expected_sources: list[str] = case_meta.get("expected_sources", [])
        tags: list[str] = case_meta.get("tags", [])

        # Cases with no expected_sources are adversarial/negative probes
        # (e.g. "Tesla Q3 2025", "Q1 2000 Amazon", "Datadog dividend") where
        # the correct answer is "no data available". Their metrics are recorded
        # in case_details for visibility but excluded from aggregate scores so
        # they don't deflate precision/recall/hit on positive cases.
        is_negative = not expected_sources

        case_result: dict[str, Any] = {
            "query": query,
            "expected_sources": expected_sources,
            "tags": tags,
            "is_negative_case": is_negative,
            "results_by_mode": {},
        }

        # Build the retrieval plan once per case: resolves entity/temporal scope,
        # rewrites the query, and pre-computes embeddings. Reused across all modes
        # so scope resolution and query rewriting run only once per case.
        scope, rewritten_queries, embedding_map, filter_metadata = await build_retrieval_plan(
            query=query,
            conversation_context=conversation_context,
            companies=companies,
            available_periods=available_periods,
        )

        for mode in modes:
            chunks = await retrieve_from_plan(
                scope=scope,
                rewritten_queries=rewritten_queries,
                embedding_map=embedding_map,
                filter_metadata=filter_metadata,
                search_mode=mode,
            )
            prec, rec, mrr, hit = await _compute_metrics(
                query, chunks, expected_sources
            )
            if not is_negative:
                mode_scores[mode].append((prec, rec, mrr, hit))
                for tag in tags:
                    if tag not in tag_mode_scores:
                        tag_mode_scores[tag] = {m: [] for m in modes}
                    tag_mode_scores[tag][mode].append((prec, rec, mrr, hit))
            mode_result: dict[str, Any] = {
                "num_returned": len(chunks),
                "precision": prec,
                "recall": rec,
                "mrr": mrr,
                "hit": hit,
            }
            if save_chunks:
                relevance_flags = [await _is_chunk_relevant(query, c, expected_sources) for c in chunks]
                mode_result["chunks"] = [
                    _chunk_to_serializable(c, rel)
                    for c, rel in zip(chunks, relevance_flags)
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
            precision=sum(x[0] for x in lst) / n,
            recall=sum(x[1] for x in lst) / n,
            mrr=sum(x[2] for x in lst) / n,
            hit=sum(x[3] for x in lst) / n,
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
        scores_by_mode=scores_by_mode,
        scores_by_tag=scores_by_tag,
        case_details=case_details,
    )

    n_positive = sum(1 for c in case_details if not c.get("is_negative_case"))
    n_negative = len(case_details) - n_positive
    logger.info(
        f"Retrieval eval {run_id} complete: {n_positive} positive cases, {n_negative} negative skipped. "
        f"vector={scores_by_mode.get('vector')}, "
        f"keyword={scores_by_mode.get('keyword')}, "
        f"hybrid={scores_by_mode.get('hybrid')}"
    )
    return result


def _mode_scores_to_dict(s: RetrievalModeScores) -> dict[str, float]:
    return {
        "precision": s.precision,
        "recall": s.recall,
        "mrr": s.mrr,
        "hit": s.hit,
    }


def retrieval_eval_to_dict(result: RetrievalEvalResult) -> dict[str, Any]:
    """Convert RetrievalEvalResult to API-serializable dict."""
    n_positive = sum(1 for c in result.case_details if not c.get("is_negative_case"))
    n_negative = len(result.case_details) - n_positive
    out: dict[str, Any] = {
        "run_id": result.run_id,
        "dataset_name": result.dataset_name,
        "n_positive_cases": n_positive,
        "n_negative_cases": n_negative,
        "scores_by_mode": {
            mode: _mode_scores_to_dict(s)
            for mode, s in result.scores_by_mode.items()
        },
        "case_details": result.case_details,
    }
    if result.scores_by_tag:
        tag_case_counts: dict[str, int] = {}
        for c in result.case_details:
            if not c.get("is_negative_case"):
                for tag in c.get("tags", []):
                    tag_case_counts[tag] = tag_case_counts.get(tag, 0) + 1
        out["scores_by_tag"] = {
            tag: {
                "n_cases": tag_case_counts.get(tag, 0),
                **{mode: _mode_scores_to_dict(s) for mode, s in tag_modes.items()},
            }
            for tag, tag_modes in result.scores_by_tag.items()
        }
    return out
