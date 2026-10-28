"""Persist analysis sessions, original predictions and append-only feedback."""
from alembic import op
import sqlalchemy as sa

revision = '0001_history'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('analysis_sessions',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('source', sa.String(16), nullable=False),
        sa.Column('filename', sa.Text(), nullable=False),
        sa.Column('created_at', sa.String(40), nullable=False),
        sa.Column('started_at', sa.String(40), nullable=False),
        sa.Column('ended_at', sa.String(40)),
        sa.Column('status', sa.String(16), nullable=False),
        sa.Column('processing_time_seconds', sa.Float()),
        sa.Column('transcript', sa.Text(), nullable=False),
        sa.Column('error', sa.Text()),
        sa.Column('model_version', sa.String(100), nullable=False),
        sa.Column('checkpoint', sa.Text(), nullable=False),
        sa.Column('classification_mode', sa.String(20), nullable=False),
        sa.Column('whisper_segment_count', sa.Integer(), nullable=False),
        sa.CheckConstraint("source IN ('upload','recording')", name='session_source'),
        sa.CheckConstraint("status IN ('processing','completed','failed')", name='session_status'))
    op.create_index('ix_sessions_created_id', 'analysis_sessions', ['created_at', 'id'])
    op.create_index('ix_sessions_status', 'analysis_sessions', ['status'])
    op.create_table('analysis_segments',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('session_id', sa.String(36), sa.ForeignKey('analysis_sessions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('position', sa.Integer(), nullable=False),
        sa.Column('unit_id', sa.String(100), nullable=False),
        sa.Column('source_segment_id', sa.JSON(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('start', sa.Float(), nullable=False),
        sa.Column('end', sa.Float(), nullable=False),
        sa.Column('timestamp_is_approximate', sa.Boolean(), nullable=False),
        sa.Column('predicted_label', sa.String(30), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('probabilities', sa.JSON(), nullable=False),
        sa.Column('explanation', sa.Text(), nullable=False),
        sa.CheckConstraint("predicted_label IN ('KHONG_CANH_BAO','CANH_BAO')", name='segment_label'),
        sa.CheckConstraint('confidence >= 0 AND confidence <= 1', name='segment_confidence'))
    op.create_index('ix_segments_session_position', 'analysis_segments', ['session_id', 'position'], unique=True)
    op.create_index('ix_segments_label_session', 'analysis_segments', ['predicted_label', 'session_id'])
    op.create_table('feedback',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('segment_id', sa.String(36), sa.ForeignKey('analysis_segments.id', ondelete='CASCADE'), nullable=False),
        sa.Column('label', sa.String(30), nullable=False),
        sa.Column('note', sa.Text(), nullable=False),
        sa.Column('created_at', sa.String(40), nullable=False),
        sa.CheckConstraint("label IN ('KHONG_CANH_BAO','CANH_BAO')", name='feedback_label'))
    op.create_index('ix_feedback_segment_created', 'feedback', ['segment_id', 'created_at'])


def downgrade():
    op.drop_table('feedback')
    op.drop_table('analysis_segments')
    op.drop_table('analysis_sessions')
