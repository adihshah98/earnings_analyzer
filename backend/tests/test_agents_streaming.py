"""Tests for deterministic helpers in `app.agents.streaming`."""

from app.agents.streaming import _is_greeting


class TestGreetingDetection:
    def test_detects_hello(self):
        assert _is_greeting("Hello")

    def test_detects_hi_punctuation(self):
        assert _is_greeting("Hi!")

    def test_not_greeting_when_substantive(self):
        assert not _is_greeting("Hello, what was Apple revenue in Q4?")
