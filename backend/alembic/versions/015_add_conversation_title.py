"""Add title column to conversation_messages for fast session listing without deserialization.

Revision ID: 015
Revises: 014
Create Date: 2026-03-29
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "015"
down_revision: Union[str, None] = "014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "conversation_messages",
        sa.Column("title", sa.Text, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("conversation_messages", "title")
