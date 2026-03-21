"""Promote company_ticker and call_date from JSONB metadata to real columns.

Adds indexed text columns for the two most-queried metadata fields so that
pgvector's WHERE clause can use a B-tree index instead of full table scans.

Revision ID: 009
Revises: 008
Create Date: 2026-03-21

"""

from typing import Sequence, Union

from alembic import op

revision: str = "010"
down_revision: Union[str, None] = "009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    for table in ("document_chunks", "eval_document_chunks"):
        op.execute(f"""
            ALTER TABLE {table}
            ADD COLUMN IF NOT EXISTS company_ticker text,
            ADD COLUMN IF NOT EXISTS call_date text
        """)
        op.execute(f"""
            UPDATE {table}
            SET company_ticker = metadata->>'company_ticker',
                call_date      = metadata->>'call_date'
            WHERE company_ticker IS NULL
        """)
        op.execute(f"""
            CREATE INDEX IF NOT EXISTS ix_{table}_company_date
            ON {table} (company_ticker, call_date)
        """)


def downgrade() -> None:
    for table in ("document_chunks", "eval_document_chunks"):
        op.execute(f"DROP INDEX IF EXISTS ix_{table}_company_date")
        op.execute(f"""
            ALTER TABLE {table}
            DROP COLUMN IF EXISTS company_ticker,
            DROP COLUMN IF EXISTS call_date
        """)
