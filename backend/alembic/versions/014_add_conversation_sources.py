"""Add sources JSONB column to conversation_messages for persisting cited sources.

Revision ID: 014
Revises: 013
Create Date: 2026-03-29
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "014"
down_revision: Union[str, None] = "013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "conversation_messages",
        sa.Column("sources", JSONB, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("conversation_messages", "sources")
