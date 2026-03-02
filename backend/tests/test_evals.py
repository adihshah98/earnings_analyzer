"""Tests for the eval system."""

import pytest

from app.evals.datasets import load_dataset, list_datasets
from app.evals.retrieval import (
    RetrievalEvalResult,
    RetrievalModeScores,
    _cache_key,
    _chunk_to_serializable,
    _title_matches,
    retrieval_eval_to_dict,
)
from app.models.schemas import EvalCase, EvalDataset, MetricScore


class TestEvalDataset:
    """Test eval dataset loading and validation."""

    def test_load_transcript_dataset(self):
        """Transcript eval dataset should load successfully."""
        dataset = load_dataset("acme_eval")
        assert dataset.name == "acme_eval"
        assert len(dataset.cases) > 0

    def test_dataset_cases_valid(self):
        """All cases should have required fields."""
        dataset = load_dataset("acme_eval")
        for case in dataset.cases:
            query = case.inputs.get("query", "") if isinstance(case.inputs, dict) else ""
            tags = case.metadata.get("tags", []) if isinstance(case.metadata, dict) else []
            assert query
            assert len(tags) > 0

    def test_missing_dataset_raises(self):
        """Loading a nonexistent dataset should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_dataset("nonexistent_dataset")

    def test_list_datasets(self):
        """Should list available datasets."""
        datasets = list_datasets()
        assert "acme_eval" in datasets

    def test_load_dataset_with_tag_filter(self):
        """Tag filter should restrict cases to those matching at least one tag."""
        full = load_dataset("acme_eval")
        filtered = load_dataset("acme_eval", tag_filter=["AI"])
        assert len(filtered.cases) < len(full.cases)
        for case in filtered.cases:
            tags = case.metadata.get("tags", []) if isinstance(case.metadata, dict) else []
            assert "AI" in tags

    def test_load_dataset_with_tag_filter_no_match(self):
        """Tag filter with no matching tags should return empty cases."""
        filtered = load_dataset("acme_eval", tag_filter=["nonexistent_tag_xyz"])
        assert len(filtered.cases) == 0


class TestMetricScore:
    """Test metric score validation."""

    def test_valid_score(self):
        score = MetricScore(name="faithfulness", score=0.85, details="Good")
        assert score.score == 0.85

    def test_score_boundaries(self):
        MetricScore(name="test", score=0.0)
        MetricScore(name="test", score=1.0)

    def test_invalid_score_above(self):
        with pytest.raises(Exception):
            MetricScore(name="test", score=1.5)

    def test_invalid_score_below(self):
        with pytest.raises(Exception):
            MetricScore(name="test", score=-0.1)

    def test_metric_score_default_details(self):
        """MetricScore details defaults to empty string."""
        score = MetricScore(name="test", score=0.5)
        assert score.details == ""


class TestEvalCase:
    """Test EvalCase schema validation."""

    def test_valid_case(self):
        case = EvalCase(
            query="What is the revenue?",
            expected_answer="$100M",
            expected_sources=["Doc A"],
            tags=["revenue"],
        )
        assert case.query == "What is the revenue?"
        assert case.tags == ["revenue"]

    def test_minimal_case(self):
        """EvalCase allows optional fields to be omitted."""
        case = EvalCase(query="minimal")
        assert case.query == "minimal"
        assert case.expected_answer is None
        assert case.expected_sources == []
        assert case.tags == []


class TestRetrievalHelpers:
    """Test retrieval eval helper functions (no network/LLM)."""

    def test_cache_key_deterministic(self):
        k1 = _cache_key("query", "content")
        k2 = _cache_key("query", "content")
        assert k1 == k2

    def test_cache_key_different_inputs(self):
        assert _cache_key("a", "b") != _cache_key("a", "c")
        assert _cache_key("a", "b") != _cache_key("c", "b")

    def test_title_matches(self):
        chunk = {"metadata": {"title": "Doc A"}, "content": "..."}
        assert _title_matches(chunk, ["Doc A", "Doc B"]) is True
        assert _title_matches(chunk, ["Doc B"]) is False

    def test_title_matches_empty_expected_sources(self):
        chunk = {"metadata": {"title": "Doc A"}}
        assert _title_matches(chunk, []) is False

    def test_title_matches_missing_title(self):
        chunk = {"metadata": {}}
        assert _title_matches(chunk, ["Doc A"]) is False

    def test_chunk_to_serializable(self):
        chunk = {"content": "hi", "metadata": {"title": "T"}, "similarity": 0.9}
        out = _chunk_to_serializable(chunk)
        assert out["content"] == "hi"
        assert out["metadata"]["title"] == "T"
        assert out["similarity"] == 0.9
        assert "llm_relevant" not in out

    def test_chunk_to_serializable_with_relevant(self):
        chunk = {"content": "hi", "metadata": {}}
        out = _chunk_to_serializable(chunk, relevant=True)
        assert out["llm_relevant"] is True


class TestRetrievalEvalResult:
    """Test RetrievalEvalResult and retrieval_eval_to_dict."""

    def test_retrieval_eval_to_dict(self):
        scores = RetrievalModeScores(precision_at_k=0.8, recall_at_k=0.6, mrr=0.5, hit_at_k=1.0)
        result = RetrievalEvalResult(
            run_id="abc",
            dataset_name="acme_eval",
            top_k=5,
            scores_by_mode={"vector": scores, "keyword": scores, "hybrid": scores},
            scores_by_tag={"AI": {"vector": scores, "keyword": scores, "hybrid": scores}},
            case_details=[{"query": "q1", "results_by_mode": {}}],
        )
        d = retrieval_eval_to_dict(result)
        assert d["run_id"] == "abc"
        assert d["dataset_name"] == "acme_eval"
        assert d["top_k"] == 5
        assert d["scores_by_mode"]["vector"]["precision_at_k"] == 0.8
        assert "scores_by_tag" in d
        assert d["scores_by_tag"]["AI"]["vector"]["recall_at_k"] == 0.6
        assert len(d["case_details"]) == 1

    def test_retrieval_eval_to_dict_empty_scores_by_tag(self):
        """When scores_by_tag is empty, it should not be included in output."""
        result = RetrievalEvalResult(
            run_id="x",
            dataset_name="test",
            top_k=3,
            scores_by_mode={},
            scores_by_tag={},
        )
        d = retrieval_eval_to_dict(result)
        assert "scores_by_tag" not in d

    def test_retrieval_mode_scores_defaults(self):
        s = RetrievalModeScores()
        assert s.precision_at_k == 0.0
        assert s.recall_at_k == 0.0
        assert s.mrr == 0.0
        assert s.hit_at_k == 0.0


class TestCalculatorTool:
    """Test the calculator tool for correctness."""

    @pytest.mark.asyncio
    async def test_valid_expression(self):
        from app.tools.agent_tools import AgentDeps, calculate

        class MockCtx:
            deps = AgentDeps()

        result = await calculate(MockCtx(), "2 + 3")
        assert result == "5"

    @pytest.mark.asyncio
    async def test_invalid_expression_should_error(self):
        """Invalid expressions should return an error, not '0'."""
        from app.tools.agent_tools import AgentDeps, calculate

        class MockCtx:
            deps = AgentDeps()

        result = await calculate(MockCtx(), "hello + world")
        assert result != "0", "Calculator should not silently return '0' for invalid input"
