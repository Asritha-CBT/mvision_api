"""make category_id not nullable

Revision ID: 82abb85cf3c0
Revises: 2b352a24aed1
Create Date: 2025-12-23 17:00:44.147827

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import text


# revision identifiers, used by Alembic.
revision: str = '82abb85cf3c0'
down_revision: Union[str, Sequence[str], None] = '2b352a24aed1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # ensure default category exists
    op.execute(text("""
        INSERT INTO camera_category (id, name)
        VALUES (1, 'Default Category')
        ON CONFLICT (id) DO NOTHING
    """))

    # update NULLs
    op.execute(text("UPDATE users SET category_id = 1 WHERE category_id IS NULL"))

    # now enforce NOT NULL
    op.alter_column(
        'users',
        'category_id',
        existing_type=sa.Integer(),
        nullable=False
    )


def downgrade() -> None:
    """Downgrade schema."""
    pass
