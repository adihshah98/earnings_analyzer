"""Add eval_document_chunks table for isolated eval transcripts.

Same schema as document_chunks; evals use this table to avoid polluting prod data.

Revision ID: 007
Revises: 006
Create Date: 2025-01-01

"""

from typing import Sequence, Union

from alembic import op

revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        create table if not exists eval_document_chunks (
            id uuid default gen_random_uuid() primary key,
            content text not null,
            metadata jsonb default '{}',
            embedding vector(1536),
            source_doc_id text,
            chunk_index integer,
            created_at timestamptz default now()
        )
    """)
    op.execute("""
        alter table eval_document_chunks
        add column content_tsv tsvector
        generated always as (to_tsvector('english', coalesce(content, ''))) stored
    """)
    op.execute("SET maintenance_work_mem = '64MB'")
    op.execute("""
        create index if not exists ix_eval_document_chunks_embedding
        on eval_document_chunks
        using ivfflat (embedding vector_cosine_ops)
        with (lists = 10)
    """)
    op.execute("""
        create index if not exists ix_eval_document_chunks_content_tsv
        on eval_document_chunks using gin (content_tsv)
    """)


def downgrade() -> None:
    op.execute("drop index if exists ix_eval_document_chunks_content_tsv")
    op.execute("drop index if exists ix_eval_document_chunks_embedding")
    op.execute("drop table if exists eval_document_chunks")
