"""Explicit approval, versioned review and immutable binary CSV snapshots.

This module never writes development splits or invokes a model. Feedback remains
evidence; only an explicit approval makes the current sample revision exportable.
"""
import csv
import hashlib
import io
import os
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from backend.database import (ReviewSample, ReviewRevision, ReviewExport, Feedback,
                              SupplementTrainingUsage,
                              AnalysisSegment, AnalysisSession, ROOT, utc_now, new_id)

EXPORT_ROOT = ROOT / 'runtime' / 'data-exports'
FIELDS = ('id','feedback_id','segment_id','session_id','source_sample_id','original_text',
          'predicted_label','probabilities','model_version','checkpoint','text','label',
          'situation','reason','family_id','source_type','status','reviewer','reviewed_at',
          'revision','created_at','updated_at')


def normalized_hash(text):
    normalized = ' '.join(unicodedata.normalize('NFC', text).casefold().split())
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def sample_dict(row):
    return {key: getattr(row, key) for key in FIELDS}


def required(db, sample_id, lock=False):
    query = select(ReviewSample).where(ReviewSample.id == sample_id)
    row = db.scalar(query.with_for_update() if lock else query)
    if row is None:
        raise HTTPException(404, 'Không tìm thấy mẫu dữ liệu bổ sung.')
    return row


def audit(db, row, action, actor=None):
    db.flush()
    db.add(ReviewRevision(sample_id=row.id, revision=row.revision, action=action,
                          actor=actor or None, snapshot=sample_dict(row)))


def check_duplicate(db, text, label, exclude=None):
    row = db.scalar(select(ReviewSample).where(ReviewSample.normalized_hash == normalized_hash(text)))
    if row and row.id != exclude:
        message = 'Xung đột nhãn với mẫu' if row.label != label else 'Văn bản đã có trong mẫu'
        raise HTTPException(409, f'{message} {row.id}. Mở mẫu hiện có để rà soát; không tạo bản sao.')


def from_feedback(database, feedback_id):
    try:
        with database.sessions.begin() as db:
            old = db.scalar(select(ReviewSample).where(ReviewSample.feedback_id == feedback_id))
            if old:
                return sample_dict(old)
            feedback = db.get(Feedback, feedback_id)
            if feedback is None:
                raise HTTPException(404, 'Không tìm thấy feedback nguồn.')
            segment = db.get(AnalysisSegment, feedback.segment_id)
            session = db.get(AnalysisSession, segment.session_id)
            text = feedback.corrected_text or segment.text
            check_duplicate(db, text, feedback.label)
            row = ReviewSample(feedback_id=feedback.id, segment_id=segment.id, session_id=session.id,
                original_text=segment.text, predicted_label=segment.predicted_label,
                probabilities=dict(segment.probabilities), model_version=session.model_version,
                checkpoint=session.checkpoint, text=text, normalized_hash=normalized_hash(text),
                label=feedback.label, situation=feedback.situation or 'CHUA_PHAN_LOAI',
                reason=feedback.note or 'Chờ bổ sung lý do duyệt',
                # All utterances from one recording conservatively share a split group.
                family_id='SESSION_' + session.id, source_type='USER_FEEDBACK')
            db.add(row)
            audit(db, row, 'CREATE_FROM_FEEDBACK')
            return sample_dict(row)
    except IntegrityError:
        with database.sessions() as db:
            old = db.scalar(select(ReviewSample).where(ReviewSample.feedback_id == feedback_id))
            if old:
                return sample_dict(old)
        raise HTTPException(409, 'Nguồn hoặc văn bản đã tồn tại; tải lại danh sách để rà soát.') from None


def create(database, payload):
    try:
        with database.sessions.begin() as db:
            check_duplicate(db, payload.text, payload.label)
            if payload.source_sample_id:
                parent = required(db, payload.source_sample_id)
                if payload.family_id != parent.family_id:
                    raise HTTPException(409, 'Biến thể của mẫu nguồn phải giữ cùng family.')
            row = ReviewSample(**payload.model_dump(), normalized_hash=normalized_hash(payload.text))
            db.add(row)
            audit(db, row, 'CREATE_PROPOSAL')
            return sample_dict(row)
    except IntegrityError:
        raise HTTPException(409, 'ID hoặc văn bản đã tồn tại. Tải lại để rà soát.') from None


def detail(database, sample_id):
    with database.sessions() as db:
        row = required(db, sample_id)
        revisions = db.scalars(select(ReviewRevision).where(ReviewRevision.sample_id == row.id)
                              .order_by(ReviewRevision.revision)).all()
        return dict(**sample_dict(row), history=[dict(revision=r.revision, action=r.action,
            actor=r.actor, created_at=r.created_at, snapshot=r.snapshot) for r in revisions])


def list_samples(database, page, page_size, status=None, label=None, situation=None):
    with database.sessions() as db:
        query = select(ReviewSample)
        for field, value in [('status',status), ('label',label), ('situation',situation)]:
            if value:
                query = query.where(getattr(ReviewSample, field) == value)
        total = db.scalar(select(func.count()).select_from(query.subquery()))
        rows = db.scalars(query.order_by(ReviewSample.updated_at.desc(), ReviewSample.id)
            .offset((page-1)*page_size).limit(page_size)).all()
        situations = db.scalars(select(ReviewSample.situation).distinct().order_by(ReviewSample.situation)).all()
        counts = dict(db.execute(select(ReviewSample.status,func.count()).group_by(ReviewSample.status)).all())
        return dict(items=[sample_dict(r) for r in rows], total=total, page=page, page_size=page_size,
                    situations=situations, counts={s:counts.get(s,0) for s in ('CHO_DUYET','DA_DUYET')},
                    accumulation=accumulation(db))


def accumulation(db):
    """Only exact approved revisions used in TRAIN count; validation is separate."""
    usages = db.execute(select(ReviewRevision.sample_id, ReviewRevision.revision,
        SupplementTrainingUsage.split).join(SupplementTrainingUsage,
        SupplementTrainingUsage.revision_id == ReviewRevision.id)).all()
    trained = {(s, r) for s, r, split in usages if split == 'train'}
    validated = {(s, r) for s, r, split in usages if split == 'validation'}
    exported = {(m['id'], m['revision']) for manifest in db.scalars(select(ReviewExport.manifest))
                for m in manifest}
    approved = db.scalars(select(ReviewSample).where(ReviewSample.status == 'DA_DUYET')).all()
    unused = [s for s in approved if (s.id, s.revision) not in trained]
    def counts(field):
        return dict(sorted(Counter(getattr(s, field) for s in unused).items()))
    sources = Counter('REAL_CONFIRMED' if s.source_type == 'USER_CONFIRMED_SPEECH'
                      else 'AI_GENERATED' if s.source_type == 'AI_GENERATED'
                      else 'OTHER_NOT_CONFIRMED_REAL' for s in unused)
    return dict(approved_count=len(approved), unused_training_count=len(unused),
        trained_current_revision_count=len(approved)-len(unused),
        exported_current_revision_count=sum((s.id,s.revision) in exported for s in approved),
        validation_only_count=sum((s.id,s.revision) in validated for s in unused),
        family_count=len(counts('family_id')), families=counts('family_id'),
        situations=counts('situation'), labels=counts('label'), sources=dict(sources))


def edit(database, sample_id, payload):
    try:
        with database.sessions.begin() as db:
            row = required(db, sample_id, lock=True)
            values = payload.model_dump(exclude={'expected_revision','editor'})
            # Retrying the exact last update is harmless, but stale edits cannot overwrite.
            if all(getattr(row,k)==v for k,v in values.items()) and payload.expected_revision in (row.revision, row.revision-1):
                return sample_dict(row)
            if payload.expected_revision != row.revision:
                raise HTTPException(409, 'Mẫu đã thay đổi ở nơi khác; tải lại trước khi sửa.')
            check_duplicate(db, payload.text, payload.label, exclude=row.id)
            if row.source_sample_id and payload.family_id != required(db,row.source_sample_id).family_id:
                raise HTTPException(409, 'Giữ biến thể cùng family với mẫu nguồn.')
            if payload.family_id != row.family_id and db.scalar(select(ReviewSample.id).where(ReviewSample.source_sample_id==row.id)):
                raise HTTPException(409, 'Mẫu có biến thể liên kết; không đổi riêng family.')
            for key, value in values.items():
                setattr(row,key,value)
            row.normalized_hash = normalized_hash(row.text)
            row.status = 'CHO_DUYET'
            row.reviewer = row.reviewed_at = None
            if row.source_type == 'USER_CONFIRMED_SPEECH':
                row.source_type = 'USER_FEEDBACK'
            row.revision += 1
            row.updated_at = utc_now()
            audit(db,row,'EDIT_REQUIRES_APPROVAL',payload.editor.strip())
            return sample_dict(row)
    except IntegrityError:
        raise HTTPException(409, 'Văn bản hoặc phiên bản bị trùng; tải lại để rà soát.') from None


def approve(database, sample_id, payload):
    with database.sessions.begin() as db:
        row = required(db,sample_id,lock=True)
        if row.status=='DA_DUYET' and row.reviewer==payload.reviewer and payload.expected_revision in (row.revision,row.revision-1):
            return sample_dict(row)
        if payload.expected_revision != row.revision:
            raise HTTPException(409, 'Mẫu đã thay đổi; phải đọc lại phiên bản mới trước khi duyệt.')
        row.status = 'DA_DUYET'
        row.reviewer = payload.reviewer
        row.reviewed_at = row.updated_at = utc_now()
        if row.source_type == 'USER_FEEDBACK':
            row.source_type = 'USER_CONFIRMED_SPEECH'
        row.revision += 1
        audit(db,row,'APPROVE',payload.reviewer)
        return sample_dict(row)


def export_dict(row):
    return dict(id=row.id, filename=row.filename, sha256=row.sha256, created_at=row.created_at,
        sample_count=len(row.manifest), manifest=row.manifest,
        local_path=str(EXPORT_ROOT / row.filename))


def materialize(row):
    """DB snapshot is canonical; atomic local copy can be rebuilt after restart."""
    data = row.content.encode('utf-8')
    if hashlib.sha256(data).hexdigest()!=row.sha256:
        raise RuntimeError('Export integrity check failed')
    EXPORT_ROOT.mkdir(parents=True,exist_ok=True)
    target = EXPORT_ROOT / row.filename
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest()!=row.sha256:
            raise HTTPException(409,'Bản CSV trên đĩa đã bị sửa. Giữ file để kiểm tra; không ghi đè.')
        return data
    handle, temporary = tempfile.mkstemp(dir=EXPORT_ROOT, suffix='.tmp')
    try:
        with os.fdopen(handle,'wb') as output:
            output.write(data)
        os.replace(temporary,target)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
    return data


def make_export(database):
    with database.sessions.begin() as db:
        # Lock rows so an edit cannot race with snapshot creation on PostgreSQL.
        rows = db.scalars(select(ReviewSample).where(ReviewSample.status=='DA_DUYET')
                          .order_by(ReviewSample.id).with_for_update()).all()
        if not rows:
            raise HTTPException(400,'Chưa có mẫu đã duyệt để xuất.')
        output = io.StringIO(newline='')
        columns = ('id','text','label','family_id','category','reason','review_status','reviewer','source',
            'original_text','predicted_label','probabilities','model_version','checkpoint','session_id',
            'segment_id','feedback_id','source_sample_id','reviewed_at','revision')
        writer = csv.DictWriter(output,fieldnames=columns,lineterminator='\n')
        writer.writeheader()
        for row in rows:
            values = sample_dict(row)
            values.update(category=row.situation,review_status=row.status,source=row.source_type,
                          probabilities=json_text(row.probabilities))
            writer.writerow({k:values.get(k) for k in columns})
        content = output.getvalue()
        digest = hashlib.sha256(content.encode('utf-8')).hexdigest()
        row = db.scalar(select(ReviewExport).where(ReviewExport.sha256==digest))
        if row is None:
            identifier = new_id()
            stamp = utc_now().replace(':','').replace('-','').replace('.','').split('+')[0]
            row = ReviewExport(id=identifier,filename=f'supplement_{stamp}_{identifier[:8]}.csv',
                sha256=digest,content=content,manifest=[dict(id=s.id,revision=s.revision,
                family_id=s.family_id,text_sha256=s.normalized_hash) for s in rows])
            db.add(row);db.flush()
    materialize(row)
    return export_dict(row)


def json_text(value):
    import json
    return json.dumps(value,ensure_ascii=False,sort_keys=True) if value is not None else ''


def list_exports(database):
    with database.sessions() as db:
        return [export_dict(r) for r in db.scalars(select(ReviewExport).order_by(ReviewExport.created_at.desc())).all()]


def get_export(database, export_id):
    with database.sessions() as db:
        row = db.get(ReviewExport,export_id)
        if row is None:
            raise HTTPException(404,'Không tìm thấy bản xuất.')
        return materialize(row),row.filename
