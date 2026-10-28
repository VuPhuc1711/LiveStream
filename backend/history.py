"""Transactions and queries for saved analyses; never changes original predictions."""
from collections import Counter
from fastapi import HTTPException
from sqlalchemy import func, select
from backend.database import AnalysisSession, AnalysisSegment, AnalysisWindow, Feedback, utc_now


def start_session(database, filename, source, health):
    with database.sessions.begin() as db:
        row = AnalysisSession(filename=filename, source=source, started_at=utc_now(),
            status='processing', model_version=health['model_version'], checkpoint=health['checkpoint'],
            classification_mode=health.get('classification_mode', 'binary'))
        db.add(row)
        db.flush()
        return row.id


def complete_session(database, session_id, result):
    with database.sessions.begin() as db:
        row = db.get(AnalysisSession, session_id)
        row.status = 'completed'
        row.ended_at = utc_now()
        row.processing_time_seconds = result.processing_time_seconds
        row.transcript = result.transcript
        row.whisper_segment_count = result.summary.whisper_segment_count
        for trace in result.asr_windows:
            db.add(AnalysisWindow(session_id=session_id, window_id=trace['window_id'],
                trace=dict(trace, session_id=session_id)))
        for index, unit in enumerate(result.classification_units):
            segment = AnalysisSegment(session_id=session_id, position=index, unit_id=unit.unit_id,
                source_segment_id=unit.source_segment_id, text=unit.text, start=unit.start, end=unit.end,
                timestamp_is_approximate=unit.timestamp_is_approximate, predicted_label=unit.final_label,
                confidence=unit.phobert_result.confidence, probabilities=unit.phobert_result.probabilities,
                explanation=unit.explanation)
            db.add(segment)
            db.flush()
            unit.segment_id = segment.id


def fail_session(database, session_id, message, elapsed):
    with database.sessions.begin() as db:
        row = db.get(AnalysisSession, session_id)
        row.status = 'failed'
        row.error = message
        row.ended_at = utc_now()
        row.processing_time_seconds = elapsed


def append_live_result(database, session_id, units, whisper_count, elapsed, trace=None):
    """One transaction per finalized batch; original predictions are append-only.

    Position and unit_id are deterministic within a live session. Repeated identical
    batches are idempotent; conflicting retries fail rather than overwrite history.
    """
    with database.sessions.begin() as db:
        row = db.get(AnalysisSession, session_id)
        if row.status != 'processing':
            raise ValueError('Session is no longer processing')
        if trace is not None:
            old_trace = db.scalar(select(AnalysisWindow).where(
                AnalysisWindow.session_id == session_id, AnalysisWindow.window_id == trace['window_id']))
            if old_trace and old_trace.trace != trace:
                raise ValueError('Conflicting window trace')
            if not old_trace:
                db.add(AnalysisWindow(session_id=session_id, window_id=trace['window_id'], trace=trace))
        added = []
        for unit in units:
            position = int(unit['unit_id'].rsplit('_', 1)[1])
            old = db.scalar(select(AnalysisSegment).where(
                AnalysisSegment.session_id == session_id, AnalysisSegment.position == position))
            values = dict(unit_id=unit['unit_id'], source_segment_id=unit['source_segment_id'],
                text=unit['text'], start=unit['start'], end=unit['end'],
                timestamp_is_approximate=unit['timestamp_is_approximate'],
                predicted_label=unit['final_label'], confidence=unit['phobert_result']['confidence'],
                probabilities=unit['phobert_result']['probabilities'], explanation=unit['explanation'])
            if old:
                if any(getattr(old, key) != value for key, value in values.items()):
                    raise ValueError('Conflicting live result')
                unit['segment_id'] = old.id
                continue
            segment = AnalysisSegment(session_id=session_id, position=position, **values)
            db.add(segment)
            db.flush()
            unit['segment_id'] = segment.id
            added.append(unit['text'])
        if added:
            row.transcript = ' '.join(filter(None, [row.transcript, *added]))
            row.whisper_segment_count += whisper_count
        row.processing_time_seconds = elapsed


def finish_live_session(database, session_id, elapsed):
    with database.sessions.begin() as db:
        row = db.get(AnalysisSession, session_id)
        row.status = 'completed'
        row.ended_at = utc_now()
        row.processing_time_seconds = elapsed


def feedback_dict(row):
    return {key: getattr(row, key) for key in ('id', 'segment_id', 'label', 'note', 'corrected_text', 'situation', 'created_at')}


def session_dict(row):
    return {key: getattr(row, key) for key in ('id', 'source', 'filename', 'created_at',
        'started_at', 'ended_at', 'status', 'processing_time_seconds', 'error', 'model_version',
        'checkpoint', 'classification_mode')}


def session_detail(database, session_id):
    with database.sessions() as db:
        row = db.get(AnalysisSession, session_id)
        if row is None:
            raise HTTPException(404, 'Không tìm thấy phiên phân tích.')
        segments = db.scalars(select(AnalysisSegment).where(AnalysisSegment.session_id == session_id)
                              .order_by(AnalysisSegment.position)).all()
        feedback = db.scalars(select(Feedback).join(AnalysisSegment)
            .where(AnalysisSegment.session_id == session_id).order_by(Feedback.created_at, Feedback.id)).all()
        by_segment = {}
        for item in feedback:
            by_segment.setdefault(item.segment_id, []).append(feedback_dict(item))
        units = [dict(segment_id=s.id, unit_id=s.unit_id, source_segment_id=s.source_segment_id,
            text=s.text, start=s.start, end=s.end, timestamp_is_approximate=s.timestamp_is_approximate,
            final_label=s.predicted_label, final_confidence=s.confidence,
            phobert_result=dict(label=s.predicted_label, confidence=s.confidence, probabilities=s.probabilities),
            explanation=s.explanation, feedback=by_segment.get(s.id, [])) for s in segments]
        counts = Counter(s.predicted_label for s in segments)
        traces = db.scalars(select(AnalysisWindow).where(AnalysisWindow.session_id == session_id)).all()
        ordered_traces = [w.trace for w in sorted(traces, key=lambda w: int(w.window_id.rsplit('_',1)[1]))]
        return dict(**session_dict(row), session_id=row.id, transcript=row.transcript,
            asr_quality_warnings=[dict(trace['asr_quality_warning'], window_id=trace['window_id'])
                for trace in ordered_traces if trace.get('asr_quality_warning')],
            asr_unresolved_windows=sum(bool(trace.get('asr_quality_warning')) for trace in ordered_traces),
            asr_recovered_windows=sum((trace.get('asr_recovery') or {}).get('state')=='recovered' for trace in ordered_traces),
            live_windows=ordered_traces if row.source in ('microphone_live', 'browser_tab_live') else [],
            asr_windows=ordered_traces,
            summary=dict(whisper_segment_count=row.whisper_segment_count, classification_unit_count=len(units),
                         final_label_counts={label: counts[label] for label in ('KHONG_CANH_BAO','CANH_BAO')}),
            classification_units=units)


def list_sessions(database, page, page_size, status, warnings_only):
    with database.sessions() as db:
        query = select(AnalysisSession)
        if status:
            query = query.where(AnalysisSession.status == status)
        if warnings_only:
            query = query.where(select(AnalysisSegment.id).where(
                AnalysisSegment.session_id == AnalysisSession.id,
                AnalysisSegment.predicted_label == 'CANH_BAO').exists())
        total = db.scalar(select(func.count()).select_from(query.subquery()))
        rows = db.scalars(query.order_by(AnalysisSession.created_at.desc(), AnalysisSession.id.desc())
            .offset((page - 1) * page_size).limit(page_size)).all()
        ids = [r.id for r in rows]
        counts = {}
        if ids:
            for sid, label, count in db.execute(select(AnalysisSegment.session_id, AnalysisSegment.predicted_label,
                func.count()).where(AnalysisSegment.session_id.in_(ids)).group_by(
                    AnalysisSegment.session_id, AnalysisSegment.predicted_label)):
                counts.setdefault(sid, {})[label] = count
        items = [dict(**session_dict(row), warning_count=counts.get(row.id, {}).get('CANH_BAO', 0),
                      segment_count=sum(counts.get(row.id, {}).values())) for row in rows]
        return dict(items=items, total=total, page=page, page_size=page_size)


def add_feedback(database, segment_id, payload):
    with database.sessions.begin() as db:
        if db.get(AnalysisSegment, segment_id) is None:
            raise HTTPException(404, 'Không tìm thấy đoạn phân tích.')
        text = payload.corrected_text.strip() if payload.corrected_text is not None else None
        if text == '':
            raise HTTPException(422, 'Văn bản sửa không được rỗng.')
        values = dict(segment_id=segment_id, label=payload.label, note=payload.note.strip(),
                      corrected_text=text, situation=payload.situation.strip())
        old = db.scalar(select(Feedback).filter_by(**values))
        if old:
            return feedback_dict(old)
        row = Feedback(**values)
        db.add(row)
        db.flush()
        return feedback_dict(row)
