"""Store raw ASR/window diagnostics without retaining audio by default."""
from alembic import op
import sqlalchemy as sa

revision = '0003_live_trace'
down_revision = '0002_live_source'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('analysis_windows',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('session_id', sa.String(36), sa.ForeignKey('analysis_sessions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('window_id', sa.String(100), nullable=False),
        sa.Column('trace', sa.JSON(), nullable=False))
    op.create_index('ix_windows_session_window', 'analysis_windows', ['session_id','window_id'], unique=True)


def downgrade():
    if op.get_bind().scalar(sa.text('SELECT count(*) FROM analysis_windows')):
        raise RuntimeError('Live traces exist; refusing to erase them.')
    op.drop_table('analysis_windows')
