from alembic import op
import sqlalchemy as sa

revision = "bb83b27c8513"
down_revision = "2dbf474a7a8b"
branch_labels = None
depends_on = None


def upgrade():
    # 1) add camera_id nullable first (table already has rows)
    op.add_column("user_presence", sa.Column("camera_id", sa.Integer(), nullable=True))

    # 2) backfill using user_presence.cam_number -> cameras.camera_name
    op.execute(sa.text("""
        UPDATE user_presence up
        SET camera_id = c.id
        FROM cameras c
        WHERE up.cam_number = c.camera_name
    """))

    # 3) rows that couldn't map (cam_number not found in cameras) -> delete OR set default
    op.execute(sa.text("DELETE FROM user_presence WHERE camera_id IS NULL"))

    # 4) now enforce NOT NULL
    op.alter_column("user_presence", "camera_id", nullable=False)

    # 5) add FK constraint
    op.create_foreign_key(
        "fk_user_presence_camera_id",
        "user_presence",
        "cameras",
        ["camera_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # 6) drop old string column
    op.drop_column("user_presence", "cam_number")


def downgrade():
    # recreate old column
    op.add_column("user_presence", sa.Column("cam_number", sa.String(), nullable=True))

    # backfill old cam_number from cameras.camera_name
    op.execute(sa.text("""
        UPDATE user_presence up
        SET cam_number = c.camera_name
        FROM cameras c
        WHERE up.camera_id = c.id
    """))

    op.drop_constraint("fk_user_presence_camera_id", "user_presence", type_="foreignkey")
    op.drop_column("user_presence", "camera_id")
