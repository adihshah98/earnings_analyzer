"""Add position column to conversation_messages for deterministic ordering.

Tool-call and tool-response messages must stay in sequence for the OpenAI API.
Ordering by created_at alone can reorder messages inserted in the same transaction.

Revision ID: 008
Revises: 007
Create Date: 2025-01-01

"""

from typing import Sequence, Union

from alembic import op

revision: str = "008"
down_revision: Union[str, None] = "007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        alter table conversation_messages
        add column if not exists position integer
    """)


def downgrade() -> None:
    op.execute("""
        alter table conversation_messages
        drop column if exists position
    """)
