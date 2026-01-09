"""fix users category_id  area_definition_id

Revision ID: ee05d557272f
Revises: 03d7de4c94c5
Create Date: 2026-01-07 15:13:51.994712

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ee05d557272f'
down_revision: Union[str, Sequence[str], None] = '03d7de4c94c5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # 1. Add new column if missing
    op.execute("""
    ALTER TABLE users
    ADD COLUMN IF NOT EXISTS area_definition_id INTEGER
    """)

    # 2. Copy data
    op.execute("""
    UPDATE users
    SET area_definition_id = category_id
    WHERE area_definition_id IS NULL
    """)

    # 3. Drop old column
    op.execute("""
    ALTER TABLE users
    DROP COLUMN IF EXISTS category_id
    """)

    # 4. Add FK
    op.execute("""
    ALTER TABLE users
    ADD CONSTRAINT users_area_definition_id_fkey
    FOREIGN KEY (area_definition_id)
    REFERENCES area_definition(id)
    """)


def downgrade() -> None:
    """Downgrade schema."""
    pass
