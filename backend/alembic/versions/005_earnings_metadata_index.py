"""Add GIN index on document_chunks.metadata for fast JSONB filtering.

Revision ID: 005
Revises: 004
Create Date: 2025-01-01

"""

from typing import Sequence, Union

from alembic import op

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        create index if not exists ix_document_chunks_metadata
        on document_chunks using gin (metadata)
    """)


def downgrade() -> None:
    op.execute("drop index if exists ix_document_chunks_metadata")
