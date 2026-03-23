"""Tests for `app.evals.context` (eval vs prod chunk table selection)."""

import pytest

from app.evals.context import use_eval_chunks, use_eval_chunks_context


class TestUseEvalChunks:
    def test_default_false(self):
        assert use_eval_chunks() is False

    @pytest.mark.asyncio
    async def test_context_manager_sets_true(self):
        assert use_eval_chunks() is False
        async with use_eval_chunks_context():
            assert use_eval_chunks() is True
        assert use_eval_chunks() is False
