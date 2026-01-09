"""migrate users.category_id to area_definition_id

Revision ID: 03d7de4c94c5
Revises: 980067180fa5
Create Date: 2026-01-07 15:04:45.110421

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '03d7de4c94c5'
down_revision: Union[str, Sequence[str], None] = '980067180fa5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    with op.batch_alter_table("users") as batch_op:
        batch_op.add_column(
            sa.Column("area_definition_id", sa.Integer(), nullable=True)
        )

    op.execute("""
        UPDATE users
        SET area_definition_id = category_id
    """)

    with op.batch_alter_table("users") as batch_op:
        batch_op.alter_column(
            "area_definition_id",
            nullable=False
        )
        batch_op.drop_column("category_id")

        batch_op.create_foreign_key(
            "users_area_definition_id_fkey",
            "area_definition",
            ["area_definition_id"],
            ["id"]
        )


def downgrade():
    with op.batch_alter_table("users") as batch_op:
        batch_op.add_column(
            sa.Column("category_id", sa.Integer(), nullable=True)
        )

    op.execute("""
        UPDATE users
        SET category_id = area_definition_id
    """)

    with op.batch_alter_table("users") as batch_op:
        batch_op.alter_column(
            "category_id",
            nullable=False
        )
        batch_op.drop_column("area_definition_id")

        batch_op.create_foreign_key(
            "users_category_id_fkey",
            "camera_category",
            ["category_id"],
            ["id"]
        )
