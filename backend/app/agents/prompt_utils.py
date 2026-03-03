"""Prompt building utilities for simple RAG."""

from typing import Any


def _sanitize_for_prompt(value: str, max_len: int = 200) -> str:
    """Sanitize user-controlled values before interpolating into prompts."""
    if not value:
        return value
    sanitized = " ".join(str(value).split())
    return sanitized[:max_len] if len(sanitized) > max_len else sanitized


def format_known_tickers(companies: list[dict[str, str]]) -> str:
    """Format a list of company dicts into a prompt-friendly string."""
    if not companies:
        return "none"
    return ", ".join(f"{c['ticker']} ({c['name']})" for c in companies if c.get("ticker"))


def build_system_prompt(
    template: str,
    *,
    context: str = "",
    known_tickers: str = "",
    today_date: str = "unknown",
    user_name: str = "anonymous",
    user_role: str = "viewer",
    user_department: str = "general",
    custom_instructions: str = "",
) -> str:
    """Build the system prompt from a template and parameters."""
    user_name = _sanitize_for_prompt(user_name)
    user_role = _sanitize_for_prompt(user_role)
    user_department = _sanitize_for_prompt(user_department)
    custom_instructions = _sanitize_for_prompt(custom_instructions, max_len=500)

    try:
        return template.format(
            context=context,
            user_name=user_name,
            user_role=user_role,
            user_department=user_department,
            custom_instructions=custom_instructions,
            known_tickers=known_tickers,
            today_date=today_date,
        )
    except KeyError:
        return (
            template.replace("{context}", context)
            .replace("{known_tickers}", known_tickers)
            .replace("{today_date}", today_date)
        )
