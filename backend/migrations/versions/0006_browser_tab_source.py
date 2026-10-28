"""Preserve existing history while allowing browser-tab monitoring sessions."""
from alembic import op
import sqlalchemy as sa

revision = '0006_browser_tab_source'
down_revision = '0005_training_usage'
branch_labels = depends_on = None


def upgrade():
    # Database.migrate handles SQLite FK enforcement around parent-table rebuilds.
    with op.batch_alter_table('analysis_sessions') as batch:
        batch.drop_constraint('session_source', type_='check')
        batch.create_check_constraint('session_source',
            "source IN ('upload','recording','microphone_live','browser_tab_live')")


def downgrade():
    if op.get_bind().scalar(sa.text(
            "SELECT count(*) FROM analysis_sessions WHERE source='browser_tab_live'")):
        raise RuntimeError('Browser-tab history exists; refusing destructive downgrade.')
    with op.batch_alter_table('analysis_sessions') as batch:
        batch.drop_constraint('session_source', type_='check')
        batch.create_check_constraint('session_source',
            "source IN ('upload','recording','microphone_live')")
