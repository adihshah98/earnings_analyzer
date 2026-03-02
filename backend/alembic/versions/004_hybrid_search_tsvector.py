"""Add tsvector column for full-text (BM25-style) keyword search.

Revision ID: 004
Revises: 003
Create Date: 2025-01-01

"""

from typing import Sequence, Union

from alembic import op

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        alter table document_chunks
        add column content_tsv tsvector
        generated always as (to_tsvector('english', coalesce(content, ''))) stored
    """)
    op.execute("""
        create index if not exists ix_document_chunks_content_tsv
        on document_chunks using gin (content_tsv)
    """)


def downgrade() -> None:
    op.execute("drop index if exists ix_document_chunks_content_tsv")
    op.execute("alter table document_chunks drop column if exists content_tsv")
