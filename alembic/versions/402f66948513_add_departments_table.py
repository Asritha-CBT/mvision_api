"""add departments table

Revision ID: 402f66948513
Revises: ee05d557272f
Create Date: 2026-01-07 23:30:29.365080

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '402f66948513'
down_revision: Union[str, Sequence[str], None] = 'ee05d557272f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create departments table
    op.create_table(
        'departments',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('name', sa.String(length=50), nullable=False, unique=True),
        sa.Column('status', sa.String(length=20), nullable=False, default='active')
    )
    op.create_index(op.f('ix_departments_id'), 'departments', ['id'], unique=False)

    # Add camera_ip column to cameras (nullable first)
    op.add_column('cameras', sa.Column('camera_ip', sa.String(length=45), nullable=True))
    op.create_unique_constraint(None, 'cameras', ['camera_ip'])

    # Recreate foreign key for user_presence
    op.drop_constraint(op.f('fk_user_presence_camera_id'), 'user_presence', type_='foreignkey')
    op.create_foreign_key(None, 'user_presence', 'cameras', ['camera_id'], ['id'])

    # Add department_id column to users as nullable
    op.add_column('users', sa.Column('department_id', sa.Integer(), nullable=True))

    # Backfill existing users with a default department
    op.execute("""
        INSERT INTO departments (name, status)
        VALUES ('General', 'active')
        ON CONFLICT (name) DO NOTHING
    """)
    op.execute("""
        UPDATE users
        SET department_id = (SELECT id FROM departments WHERE name='General')
        WHERE department_id IS NULL
    """)

    # Add unique constraint and foreign key now that data is present
    op.create_unique_constraint('uq_user_name_department', 'users', ['name', 'department_id'])
    op.create_foreign_key(None, 'users', 'departments', ['department_id'], ['id'])

    # Drop old columns
    op.drop_column('users', 'gender')
    op.drop_column('users', 'department')


def downgrade() -> None:
    """Downgrade schema."""
    # Add back old columns
    op.add_column('users', sa.Column('department', sa.VARCHAR(), nullable=False))
    op.add_column('users', sa.Column('gender', sa.VARCHAR(), nullable=False))

    # Drop constraints and new columns
    op.drop_constraint(None, 'users', type_='foreignkey')
    op.drop_constraint('uq_user_name_department', 'users', type_='unique')
    op.drop_column('users', 'department_id')

    op.drop_constraint(None, 'user_presence', type_='foreignkey')
    op.create_foreign_key(
        op.f('fk_user_presence_camera_id'),
        'user_presence', 'cameras',
        ['camera_id'], ['id'], ondelete='CASCADE'
    )

    op.drop_constraint(None, 'cameras', type_='unique')
    op.drop_column('cameras', 'camera_ip')

    # Drop departments table
    op.drop_index(op.f('ix_departments_id'), table_name='departments')
    op.drop_table('departments')
