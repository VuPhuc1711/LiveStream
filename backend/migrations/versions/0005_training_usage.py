"""Keep exported snapshots separate from actual versioned training usage."""
from alembic import op
import sqlalchemy as sa

revision = '0005_training_usage'
down_revision = '0004_review_samples'
branch_labels = depends_on = None


def upgrade():
    op.create_table('supplement_training_runs',
        sa.Column('id', sa.String(120), primary_key=True),
        sa.Column('checkpoint', sa.Text(), nullable=False),
        sa.Column('config_sha256', sa.String(64), nullable=False),
        sa.Column('completed_at', sa.String(40), nullable=False))
    op.create_table('supplement_training_usage',
        sa.Column('run_id', sa.String(120), sa.ForeignKey('supplement_training_runs.id'), primary_key=True),
        sa.Column('revision_id', sa.String(36), sa.ForeignKey('review_revisions.id'), primary_key=True),
        sa.Column('split', sa.String(20), nullable=False),
        sa.CheckConstraint("split IN ('train','validation')", name='supplement_usage_split'))


def downgrade():
    if op.get_bind().scalar(sa.text('SELECT count(*) FROM supplement_training_runs')):
        raise RuntimeError('Training usage exists; refusing destructive downgrade.')
    op.drop_table('supplement_training_usage')
    op.drop_table('supplement_training_runs')
