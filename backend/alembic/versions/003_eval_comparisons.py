"""Add eval_comparisons table for prompt comparison results.

Revision ID: 003
Revises: 002
Create Date: 2025-01-01

"""

from typing import Sequence, Union

from alembic import op

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        create table if not exists eval_comparisons (
            id uuid default gen_random_uuid() primary key,
            run_id text not null,
            dataset_name text not null,
            prompt_version_a text not null,
            prompt_version_b text not null,
            scores_a jsonb not null,
            scores_b jsonb not null,
            bootstrap_ci jsonb default '{}',
            details jsonb default '{}',
            created_at timestamptz default now()
        )
    """)


def downgrade() -> None:
    op.execute("drop table if exists eval_comparisons")
