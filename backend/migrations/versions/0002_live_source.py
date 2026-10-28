"""Allow live microphone sessions; existing unique segment positions remain enforced."""
from alembic import op

revision = '0002_live_source'
down_revision = '0001_history'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('analysis_sessions') as batch:
        batch.drop_constraint('session_source', type_='check')
        batch.create_check_constraint('session_source', "source IN ('upload','recording','microphone_live')")


def downgrade():
    # Refuse a destructive downgrade when live history exists.
    from sqlalchemy import text
    if op.get_bind().scalar(text("SELECT count(*) FROM analysis_sessions WHERE source='microphone_live'")):
        raise RuntimeError('Live history exists; keep the current schema.')
    with op.batch_alter_table('analysis_sessions') as batch:
        batch.drop_constraint('session_source', type_='check')
        batch.create_check_constraint('session_source', "source IN ('upload','recording')")
