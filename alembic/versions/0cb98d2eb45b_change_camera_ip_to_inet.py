"""change camera_ip to inet

Revision ID: 0cb98d2eb45b
Revises: ad77708d2c31
Create Date: 2026-01-09 12:04:21.869435

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0cb98d2eb45b'
down_revision: Union[str, Sequence[str], None] = 'ad77708d2c31'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
