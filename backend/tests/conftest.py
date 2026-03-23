"""Pytest configuration.

Test modules follow ``test_<area>_<topic>.py`` so each file maps to a slice of the codebase:

- ``test_models_*`` — Pydantic schemas in ``app.models``
- ``test_agents_*`` — ``app.agents`` (simple RAG, prompts, streaming helpers)
- ``test_rag_*`` — ``app.rag`` (ingestion, retriever, fiscal calendar)
- ``test_evals_*`` — ``app.evals`` (datasets, context, retrieval eval helpers)
"""
