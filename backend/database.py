"""SQLAlchemy persistence; timestamps stay UTC ISO-8601 across both databases."""
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from alembic import command
from alembic.config import Config
from sqlalchemy import (JSON, Boolean, CheckConstraint, Column, Float, ForeignKey,
                        Index, Integer, String, Text, create_engine, event)
from sqlalchemy.engine import make_url
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
Base = declarative_base()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def new_id():
    return str(uuid4())


class AnalysisSession(Base):
    __tablename__ = 'analysis_sessions'
    id = Column(String(36), primary_key=True, default=new_id)
    source = Column(String(16), nullable=False)
    filename = Column(Text, nullable=False)
    created_at = Column(String(40), nullable=False, default=utc_now)
    started_at = Column(String(40), nullable=False)
    ended_at = Column(String(40))
    status = Column(String(16), nullable=False)
    processing_time_seconds = Column(Float)
    transcript = Column(Text, nullable=False, default='')
    error = Column(Text)
    model_version = Column(String(100), nullable=False)
    checkpoint = Column(Text, nullable=False)
    classification_mode = Column(String(20), nullable=False)
    whisper_segment_count = Column(Integer, nullable=False, default=0)
    __table_args__ = (
        CheckConstraint("source IN ('upload','recording','microphone_live','browser_tab_live')", name='session_source'),
        CheckConstraint("status IN ('processing','completed','failed')", name='session_status'),
        Index('ix_sessions_created_id', 'created_at', 'id'),
        Index('ix_sessions_status', 'status'),
    )


class AnalysisSegment(Base):
    __tablename__ = 'analysis_segments'
    id = Column(String(36), primary_key=True, default=new_id)
    session_id = Column(String(36), ForeignKey('analysis_sessions.id', ondelete='CASCADE'), nullable=False)
    position = Column(Integer, nullable=False)
    unit_id = Column(String(100), nullable=False)
    source_segment_id = Column(JSON, nullable=False)
    text = Column(Text, nullable=False)
    start = Column(Float, nullable=False)
    end = Column(Float, nullable=False)
    timestamp_is_approximate = Column(Boolean, nullable=False)
    predicted_label = Column(String(30), nullable=False)
    confidence = Column(Float, nullable=False)
    probabilities = Column(JSON, nullable=False)
    explanation = Column(Text, nullable=False)
    __table_args__ = (
        CheckConstraint("predicted_label IN ('KHONG_CANH_BAO','CANH_BAO')", name='segment_label'),
        CheckConstraint('confidence >= 0 AND confidence <= 1', name='segment_confidence'),
        Index('ix_segments_session_position', 'session_id', 'position', unique=True),
        Index('ix_segments_label_session', 'predicted_label', 'session_id'),
    )


class AnalysisWindow(Base):
    __tablename__ = 'analysis_windows'
    id = Column(String(36), primary_key=True, default=new_id)
    session_id = Column(String(36), ForeignKey('analysis_sessions.id', ondelete='CASCADE'), nullable=False)
    window_id = Column(String(100), nullable=False)
    trace = Column(JSON, nullable=False)
    __table_args__ = (Index('ix_windows_session_window', 'session_id', 'window_id', unique=True),)


class Feedback(Base):
    __tablename__ = 'feedback'
    id = Column(String(36), primary_key=True, default=new_id)
    segment_id = Column(String(36), ForeignKey('analysis_segments.id', ondelete='CASCADE'), nullable=False)
    label = Column(String(30), nullable=False)
    note = Column(Text, nullable=False, default='')
    corrected_text = Column(Text)
    situation = Column(String(200), nullable=False, default='')
    created_at = Column(String(40), nullable=False, default=utc_now)
    __table_args__ = (
        CheckConstraint("label IN ('KHONG_CANH_BAO','CANH_BAO')", name='feedback_label'),
        Index('ix_feedback_segment_created', 'segment_id', 'created_at'),
    )


class ReviewSample(Base):
    __tablename__ = 'review_samples'
    id = Column(String(36), primary_key=True, default=new_id)
    feedback_id = Column(String(36), ForeignKey('feedback.id'), unique=True)
    segment_id = Column(String(36), ForeignKey('analysis_segments.id'))
    session_id = Column(String(36), ForeignKey('analysis_sessions.id'))
    source_sample_id = Column(String(36), ForeignKey('review_samples.id'))
    original_text = Column(Text)
    predicted_label = Column(String(30))
    probabilities = Column(JSON)
    model_version = Column(String(100))
    checkpoint = Column(Text)
    text = Column(Text, nullable=False)
    normalized_hash = Column(String(64), nullable=False, unique=True)
    label = Column(String(30), nullable=False)
    situation = Column(String(200), nullable=False)
    reason = Column(Text, nullable=False)
    family_id = Column(String(120), nullable=False)
    source_type = Column(String(40), nullable=False)
    status = Column(String(20), nullable=False, default='CHO_DUYET')
    reviewer = Column(String(120))
    reviewed_at = Column(String(40))
    revision = Column(Integer, nullable=False, default=1)
    created_at = Column(String(40), nullable=False, default=utc_now)
    updated_at = Column(String(40), nullable=False, default=utc_now)
    __table_args__ = (
        CheckConstraint("label IN ('KHONG_CANH_BAO','CANH_BAO')", name='review_sample_label'),
        CheckConstraint("status IN ('CHO_DUYET','DA_DUYET')", name='review_sample_status'),
        CheckConstraint("source_type IN ('USER_FEEDBACK','USER_CONFIRMED_SPEECH','AI_GENERATED','USER_ENTERED')", name='review_sample_source'),
        CheckConstraint("status != 'DA_DUYET' OR (reviewer IS NOT NULL AND reviewed_at IS NOT NULL)", name='review_sample_approval'),
        Index('ix_review_samples_status_label', 'status', 'label'),
        Index('ix_review_samples_session', 'session_id'),
        Index('ix_review_samples_family', 'family_id'),
    )


class ReviewRevision(Base):
    __tablename__ = 'review_revisions'
    id = Column(String(36), primary_key=True, default=new_id)
    sample_id = Column(String(36), ForeignKey('review_samples.id'), nullable=False)
    revision = Column(Integer, nullable=False)
    action = Column(String(30), nullable=False)
    actor = Column(String(120))
    snapshot = Column(JSON, nullable=False)
    created_at = Column(String(40), nullable=False, default=utc_now)
    __table_args__ = (Index('ix_review_revisions_sample_revision', 'sample_id', 'revision', unique=True),)


class ReviewExport(Base):
    __tablename__ = 'review_exports'
    id = Column(String(36), primary_key=True, default=new_id)
    filename = Column(String(120), nullable=False, unique=True)
    sha256 = Column(String(64), nullable=False, unique=True)
    content = Column(Text, nullable=False)
    manifest = Column(JSON, nullable=False)
    created_at = Column(String(40), nullable=False, default=utc_now)


class SupplementTrainingRun(Base):
    """Evidence of an actual training run, never created by CSV export."""
    __tablename__ = 'supplement_training_runs'
    id = Column(String(120), primary_key=True)
    checkpoint = Column(Text, nullable=False)
    config_sha256 = Column(String(64), nullable=False)
    completed_at = Column(String(40), nullable=False)


class SupplementTrainingUsage(Base):
    __tablename__ = 'supplement_training_usage'
    run_id = Column(String(120), ForeignKey('supplement_training_runs.id'), primary_key=True)
    revision_id = Column(String(36), ForeignKey('review_revisions.id'), primary_key=True)
    split = Column(String(20), nullable=False)
    __table_args__ = (CheckConstraint("split IN ('train','validation')", name='supplement_usage_split'),)


class Database:
    def __init__(self, url=None):
        load_dotenv(ROOT / '.env', override=False)
        self.url = url or os.environ.get('DATABASE_URL') or 'sqlite:///' + (ROOT / 'runtime' / 'livestream.sqlite3').as_posix()
        parsed = make_url(self.url)
        self.backend = parsed.get_backend_name()
        if self.backend == 'sqlite':
            if parsed.database and parsed.database != ':memory:':
                Path(parsed.database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            self.engine = create_engine(self.url, connect_args={'check_same_thread': False, 'timeout': 30})

            @event.listens_for(self.engine, 'connect')
            def configure_sqlite(connection, _):
                connection.execute('PRAGMA foreign_keys=ON')
                connection.execute('PRAGMA journal_mode=WAL')
        elif self.backend == 'postgresql':
            if parsed.drivername != 'postgresql+psycopg':
                raise ValueError('Use the postgresql+psycopg driver')
            self.engine = create_engine(self.url, pool_pre_ping=True, hide_parameters=True,
                                        connect_args={'connect_timeout': 10})
        else:
            raise ValueError('DATABASE_URL must use SQLite or PostgreSQL')
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    def health(self):
        from sqlalchemy import text
        with self.engine.connect() as connection:
            connection.execute(text('SELECT 1'))
        # Never expose credentials or a connection string in health/logs.
        return {'engine': self.backend, 'ready': True}

    def migrate(self):
        config = Config(str(ROOT / 'alembic.ini'))
        config.set_main_option('script_location', str(ROOT / 'backend' / 'migrations'))
        with self.engine.connect() as connection:
            # SQLite batch ALTER rebuilds a referenced parent table. Disable FK
            # enforcement outside the transaction so ON DELETE CASCADE cannot
            # erase existing history, then verify all relationships afterwards.
            if self.backend == 'sqlite':
                connection.exec_driver_sql('PRAGMA foreign_keys=OFF')
                connection.commit()
            try:
                with connection.begin():
                    config.attributes['connection'] = connection
                    command.upgrade(config, 'head')
            finally:
                if self.backend == 'sqlite':
                    connection.exec_driver_sql('PRAGMA foreign_keys=ON')
                    if connection.exec_driver_sql('PRAGMA foreign_key_check').fetchall():
                        raise RuntimeError('Foreign-key verification failed after migration')
                    connection.commit()

    def interrupt_processing(self):
        with self.sessions.begin() as db:
            db.query(AnalysisSession).filter_by(status='processing').update({
                'status': 'failed', 'ended_at': utc_now(),
                'error': 'Phiên bị gián đoạn khi backend dừng; không tự chạy lại.'})

    def close(self):
        self.engine.dispose()
