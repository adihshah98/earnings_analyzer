"""Add conversation_messages table for multi-turn conversation support.

Revision ID: 002
Revises: 001
Create Date: 2025-01-01

"""

from typing import Sequence, Union

from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        create table if not exists conversation_messages (
            id uuid default gen_random_uuid() primary key,
            session_id text not null,
            role text not null,
            content jsonb not null,
            created_at timestamptz default now()
        )
    """)
    op.execute("""
        create index if not exists ix_conversation_messages_session_id
        on conversation_messages (session_id)
    """)
    op.execute("""
        create index if not exists ix_conversation_messages_session_created
        on conversation_messages (session_id, created_at)
    """)


def downgrade() -> None:
    op.execute("drop table if exists conversation_messages")
