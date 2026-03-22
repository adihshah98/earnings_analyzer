"""Add fiscal_quarter column to document_chunks and eval_document_chunks.

Promotes fiscal_quarter from JSONB metadata to a dedicated column,
consistent with company_ticker and call_date.

Revision ID: 011
Revises: 010
Create Date: 2026-03-21
"""

from typing import Sequence, Union

from alembic import op

revision: str = "011"
down_revision: Union[str, None] = "010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    for table in ("document_chunks", "eval_document_chunks"):
        op.execute(f"""
            ALTER TABLE {table}
            ADD COLUMN IF NOT EXISTS fiscal_quarter text
        """)
        op.execute(f"""
            UPDATE {table}
            SET fiscal_quarter = metadata->>'fiscal_quarter'
            WHERE fiscal_quarter IS NULL
        """)


def downgrade() -> None:
    for table in ("document_chunks", "eval_document_chunks"):
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS fiscal_quarter")
