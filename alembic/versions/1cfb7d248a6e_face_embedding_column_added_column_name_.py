"""face embedding column added, column name changed from embedding tp face_embedding

Revision ID: 1cfb7d248a6e
Revises: 8c086627f54d
Create Date: 2025-12-11 16:32:17.940959

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = '1cfb7d248a6e'
down_revision: Union[str, Sequence[str], None] = '8c086627f54d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'users',
        sa.Column('body_embedding', Vector(dim=512), nullable=True)
    )
    op.add_column(
        'users',
        sa.Column('face_embedding', Vector(dim=512), nullable=True)
    )
    op.drop_column('users', 'embedding')



def downgrade() -> None:
    """Downgrade schema."""
    op.add_column(
        'users',
        sa.Column('embedding', Vector(dim=512), nullable=True)
    )
    op.drop_column('users', 'face_embedding')
    op.drop_column('users', 'body_embedding')
