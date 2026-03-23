"""Drop legacy GIN index on document_chunks.metadata JSONB column.

All hot-path filtering now uses the promoted company_ticker / call_date
B-tree columns (migration 010) so the GIN index on the raw JSONB blob is
unused overhead.

Revision ID: 013
Revises: 012
Create Date: 2026-03-22
"""

from typing import Sequence, Union

from alembic import op

revision: str = "013"
down_revision: Union[str, None] = "012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_metadata")


def downgrade() -> None:
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_document_chunks_metadata
        ON document_chunks USING gin (metadata)
    """)
