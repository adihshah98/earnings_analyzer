"""Add is_approved column to users table. Existing users are grandfathered in as approved.

Revision ID: 016
Revises: 015
Create Date: 2026-03-29
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "016"
down_revision: Union[str, None] = "015"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("is_approved", sa.Boolean, nullable=False, server_default="false"),
    )
    # Grandfather in all existing users so the owner isn't locked out
    op.execute("UPDATE users SET is_approved = TRUE")


def downgrade() -> None:
    op.drop_column("users", "is_approved")
