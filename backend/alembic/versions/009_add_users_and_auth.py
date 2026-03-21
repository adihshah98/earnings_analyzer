"""Add users table and user_id to conversation_messages for Google OAuth.

Revision ID: 009
Revises: 008
Create Date: 2026-03-20

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "009"
down_revision: Union[str, None] = "008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        create table if not exists users (
            id          uuid primary key default gen_random_uuid(),
            google_id   text not null unique,
            email       text not null unique,
            name        text not null,
            avatar_url  text,
            created_at  timestamptz default now(),
            updated_at  timestamptz default now()
        )
    """)
    op.execute("create index if not exists ix_users_google_id on users (google_id)")
    op.execute("create index if not exists ix_users_email on users (email)")

    op.execute("""
        alter table conversation_messages
        add column if not exists user_id uuid references users(id) on delete set null
    """)
    op.execute("""
        create index if not exists ix_conversation_messages_user_id
        on conversation_messages (user_id)
    """)


def downgrade() -> None:
    op.execute("drop index if exists ix_conversation_messages_user_id")
    op.execute("alter table conversation_messages drop column if exists user_id")
    op.execute("drop index if exists ix_users_email")
    op.execute("drop index if exists ix_users_google_id")
    op.execute("drop table if exists users")
