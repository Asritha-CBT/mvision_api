"""migrate users.category_id to area_definition_id

Revision ID: 980067180fa5
Revises: 5d937aed7153
Create Date: 2026-01-07 14:51:59.922307

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '980067180fa5'
down_revision: Union[str, Sequence[str], None] = '5d937aed7153'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Add new column
    op.add_column(
        'users',
        sa.Column('area_definition_id', sa.Integer(), nullable=True)
    )

    # Copy data
    op.execute("""
        UPDATE users
        SET area_definition_id = category_id
    """)

    # Enforce NOT NULL
    op.alter_column(
        'users',
        'area_definition_id',
        nullable=False
    )

    # Drop old FK + column
    op.drop_constraint('users_category_id_fkey', 'users', type_='foreignkey')
    op.drop_column('users', 'category_id')

    # Create new FK
    op.create_foreign_key(
        'users_area_definition_id_fkey',
        'users',
        'area_definition',
        ['area_definition_id'],
        ['id']
    )


def downgrade():
    op.add_column(
        'users',
        sa.Column('category_id', sa.Integer(), nullable=True)
    )

    op.execute("""
        UPDATE users
        SET category_id = area_definition_id
    """)

    op.alter_column(
        'users',
        'category_id',
        nullable=False
    )

    op.drop_constraint('users_area_definition_id_fkey', 'users', type_='foreignkey')
    op.drop_column('users', 'area_definition_id')

    op.create_foreign_key(
        'users_category_id_fkey',
        'users',
        'camera_category',
        ['category_id'],
        ['id']
    )

