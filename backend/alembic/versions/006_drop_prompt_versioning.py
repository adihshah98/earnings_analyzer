"""Drop prompt_versions and eval_comparisons tables.

Revision ID: 006
Revises: 005
Create Date: 2025-01-01

"""

from typing import Sequence, Union

from alembic import op

revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop FK and column from eval_results (if they exist) before dropping prompt_versions
    op.execute(
        "alter table eval_results drop constraint if exists eval_results_prompt_version_id_fkey"
    )
    op.execute(
        "alter table eval_results drop column if exists prompt_version_id"
    )
    op.execute("drop table if exists eval_comparisons")
    op.execute("drop table if exists prompt_versions")


def downgrade() -> None:
    # Recreate prompt_versions (from 001)
    op.execute("""
        create table if not exists prompt_versions (
            id uuid default gen_random_uuid() primary key,
            name text not null,
            version integer not null,
            template text not null,
            is_active boolean default false,
            weight float default 0.0,
            metadata jsonb default '{}',
            created_at timestamptz default now(),
            unique(name, version)
        )
    """)
    # Recreate eval_comparisons (from 003)
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
