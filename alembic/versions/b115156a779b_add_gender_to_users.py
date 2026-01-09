"""add gender to users

Revision ID: b115156a779b
Revises: bb83b27c8513
Create Date: 2026-01-06 16:07:53.175879

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b115156a779b'
down_revision: Union[str, Sequence[str], None] = 'bb83b27c8513'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # 1️⃣ Add column as nullable
    op.add_column(
        "users",
        sa.Column("gender", sa.String(), nullable=True)
    )

    # 2️⃣ Backfill existing records
    op.execute("UPDATE users SET gender = 'other' WHERE gender IS NULL")

    # 3️⃣ Enforce NOT NULL
    op.alter_column(
        "users",
        "gender",
        nullable=False
    ) 
def downgrade():
    op.drop_column("users", "gender")
