"""SQLAlchemy ORM models for database tables."""

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Computed,
    DateTime,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for SQLAlchemy models."""

    pass


class DocumentChunk(Base):
    """Document chunks for RAG storage with pgvector embeddings."""

    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
    )
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(1536),
        nullable=True,
    )
    content_tsv: Mapped[object | None] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', coalesce(content, ''))"),
        nullable=True,
        deferred=True,
    )
    source_doc_id: Mapped[str | None] = mapped_column(String, nullable=True)
    chunk_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        server_default=text("now()"),
    )


class EvalDocumentChunk(Base):
    """Eval-only document chunks. Same schema as DocumentChunk; keeps eval transcripts isolated."""

    __tablename__ = "eval_document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
    )
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(1536),
        nullable=True,
    )
    content_tsv: Mapped[object | None] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', coalesce(content, ''))"),
        nullable=True,
        deferred=True,
    )
    source_doc_id: Mapped[str | None] = mapped_column(String, nullable=True)
    chunk_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        server_default=text("now()"),
    )


class ConversationMessage(Base):
    """Conversation history for multi-turn agent support."""

    __tablename__ = "conversation_messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    session_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    role: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[list | dict[str, Any]] = mapped_column(JSONB, nullable=False)
    position: Mapped[int | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        server_default=text("now()"),
    )


class EvalResult(Base):
    """Eval run results stored for analysis."""

    __tablename__ = "eval_results"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    run_id: Mapped[str] = mapped_column(String, nullable=False)
    dataset_name: Mapped[str] = mapped_column(String, nullable=False)
    scores: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    details: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
    )
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        server_default=text("now()"),
    )
