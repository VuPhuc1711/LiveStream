"""Explicitly reviewed supplements; keep original ASR/predictions and prior feedback."""
from alembic import op
import sqlalchemy as sa

revision = '0004_review_samples'
down_revision = '0003_live_trace'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('feedback', sa.Column('corrected_text', sa.Text()))
    op.add_column('feedback', sa.Column('situation', sa.String(200), nullable=False, server_default=''))
    op.create_table('review_samples',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('feedback_id', sa.String(36), sa.ForeignKey('feedback.id'), unique=True),
        sa.Column('segment_id', sa.String(36), sa.ForeignKey('analysis_segments.id')),
        sa.Column('session_id', sa.String(36), sa.ForeignKey('analysis_sessions.id')),
        sa.Column('source_sample_id', sa.String(36), sa.ForeignKey('review_samples.id')),
        sa.Column('original_text', sa.Text()), sa.Column('predicted_label', sa.String(30)),
        sa.Column('probabilities', sa.JSON()), sa.Column('model_version', sa.String(100)),
        sa.Column('checkpoint', sa.Text()), sa.Column('text', sa.Text(), nullable=False),
        sa.Column('normalized_hash', sa.String(64), nullable=False, unique=True),
        sa.Column('label', sa.String(30), nullable=False), sa.Column('situation', sa.String(200), nullable=False),
        sa.Column('reason', sa.Text(), nullable=False), sa.Column('family_id', sa.String(120), nullable=False),
        sa.Column('source_type', sa.String(40), nullable=False), sa.Column('status', sa.String(20), nullable=False),
        sa.Column('reviewer', sa.String(120)), sa.Column('reviewed_at', sa.String(40)),
        sa.Column('revision', sa.Integer(), nullable=False), sa.Column('created_at', sa.String(40), nullable=False),
        sa.Column('updated_at', sa.String(40), nullable=False),
        sa.CheckConstraint("label IN ('KHONG_CANH_BAO','CANH_BAO')", name='review_sample_label'),
        sa.CheckConstraint("status IN ('CHO_DUYET','DA_DUYET')", name='review_sample_status'),
        sa.CheckConstraint("source_type IN ('USER_FEEDBACK','USER_CONFIRMED_SPEECH','AI_GENERATED','USER_ENTERED')", name='review_sample_source'),
        sa.CheckConstraint("status != 'DA_DUYET' OR (reviewer IS NOT NULL AND reviewed_at IS NOT NULL)", name='review_sample_approval'))
    op.create_index('ix_review_samples_status_label', 'review_samples', ['status','label'])
    op.create_index('ix_review_samples_session', 'review_samples', ['session_id'])
    op.create_index('ix_review_samples_family', 'review_samples', ['family_id'])
    op.create_table('review_revisions',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('sample_id', sa.String(36), sa.ForeignKey('review_samples.id'), nullable=False),
        sa.Column('revision', sa.Integer(), nullable=False), sa.Column('action', sa.String(30), nullable=False),
        sa.Column('actor', sa.String(120)), sa.Column('snapshot', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.String(40), nullable=False))
    op.create_index('ix_review_revisions_sample_revision','review_revisions',['sample_id','revision'],unique=True)
    op.create_table('review_exports',
        sa.Column('id',sa.String(36),primary_key=True), sa.Column('filename',sa.String(120),nullable=False,unique=True),
        sa.Column('sha256',sa.String(64),nullable=False,unique=True), sa.Column('content',sa.Text(),nullable=False),
        sa.Column('manifest',sa.JSON(),nullable=False), sa.Column('created_at',sa.String(40),nullable=False))


def downgrade():
    if op.get_bind().scalar(sa.text("SELECT count(*) FROM feedback WHERE corrected_text IS NOT NULL OR situation != ''")):
        raise RuntimeError('Corrected feedback exists; refusing destructive downgrade.')
    for name in ('review_samples','review_exports'):
        if op.get_bind().scalar(sa.text(f'SELECT count(*) FROM {name}')):
            raise RuntimeError('Review data exists; refusing destructive downgrade.')
    op.drop_table('review_revisions')
    op.drop_table('review_exports')
    op.drop_table('review_samples')
    op.drop_column('feedback','situation')
    op.drop_column('feedback','corrected_text')
