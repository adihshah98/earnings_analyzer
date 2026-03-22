"""Add period_end column to document_chunks and eval_document_chunks.

Stores the calendar date each fiscal quarter ends, enabling temporal
filtering and cross-company calendar alignment in retrieval.

Revision ID: 012
Revises: 011
Create Date: 2026-03-22
"""

from typing import Sequence, Union

from alembic import op

revision: str = "012"
down_revision: Union[str, None] = "011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    for table in ("document_chunks", "eval_document_chunks"):
        op.execute(f"""
            ALTER TABLE {table}
            ADD COLUMN IF NOT EXISTS period_end date
        """)
        op.execute(f"""
            CREATE INDEX IF NOT EXISTS ix_{table}_ticker_period
            ON {table} (company_ticker, period_end)
        """)


def downgrade() -> None:
    for table in ("document_chunks", "eval_document_chunks"):
        op.execute(f"DROP INDEX IF EXISTS ix_{table}_ticker_period")
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS period_end")
