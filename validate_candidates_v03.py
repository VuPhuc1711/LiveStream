"""Read-only structural/leakage validation; never imports or runs an ML model."""
import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = {'KHONG_CANH_BAO', 'CAN_XAC_MINH', 'CANH_BAO'}
REQUIRED = ['id', 'text', 'proposed_label', 'category', 'cue', 'reason', 'family_id',
            'variant_type', 'source', 'review_status', 'reviewer', 'review_note',
            'clean_reference_text', 'asr_noise_note', 'attention_note']
NONEMPTY = REQUIRED[:10]

def normalize(text):
    """Conservative duplicate screen, not model preprocessing."""
    return ' '.join(re.findall(r'[^\W_]+', unicodedata.normalize('NFC', text).casefold(), re.UNICODE))

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        rows = list(reader)
    if len(fields) != len(set(fields)) or any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError(f'Malformed CSV: {path}')
    return fields, rows

def historical_rows(root=ROOT):
    paths = sorted((root / 'data').rglob('*.csv'))
    result = []
    for path in paths:
        if path.name == 'candidates_v03_review.csv':
            continue
        _, rows = read_csv(path)
        for row in rows:
            if 'text' in row:
                result.append(dict(source=path.relative_to(root).as_posix(), **row))
    # Also screen against existing Whisper transcripts, not just clean test texts.
    for path in sorted((root / 'reports').glob('audio_demo_*/result.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        for key in ['whisper_segments', 'classification_units', 'segments']:
            for i, row in enumerate(data.get(key, [])):
                if row.get('text'):
                    result.append(dict(source=path.relative_to(root).as_posix() + ':' + key,
                                       id=f"ASR_{i}", text=row['text'], family_id=''))
    return result

def validate(path, review_progress=False, root=ROOT):
    fields, rows = read_csv(path)
    errors = []
    def check(condition, message):
        if not condition:
            errors.append(message)
    check(set(REQUIRED) <= set(fields), 'Thiếu cột bắt buộc')
    if errors:
        return {'status': 'INVALID', 'errors': errors}
    check(len(rows) == 60, 'Cần đúng 60 dòng')
    old = historical_rows(root)
    old_ids = {r['id'] for r in old if r.get('id')}
    old_families = {r['family_id'] for r in old if r.get('family_id')}
    old_texts = defaultdict(list)
    for row in old:
        old_texts[normalize(row['text'])].append({'source': row['source'], 'id': row.get('id')})
    families = defaultdict(list)
    labels = Counter(r['proposed_label'] for r in rows)
    variants = Counter(r['variant_type'] for r in rows)
    collisions = []
    seen_texts = {}
    check(len({r['id'] for r in rows}) == len(rows), 'ID ứng viên bị trùng')
    for row in rows:
        rid = row['id']
        check(all(row[c].strip() for c in NONEMPTY), f'{rid}: rỗng trường bắt buộc')
        check(row['proposed_label'] in LABELS, f'{rid}: nhãn không hợp lệ')
        check(row['source'] == 'TARGETED_ERROR_ANALYSIS_V03', f'{rid}: sai source')
        check(row['variant_type'] in {'CLEAN', 'ASR_LIKE'}, f'{rid}: sai variant_type')
        check(rid not in old_ids, f'{rid}: trùng ID cũ')
        check(row['family_id'] not in old_families, f'{rid}: trùng family cũ')
        if review_progress:
            check(row['review_status'] in {'CHUA_DUYET', 'DA_DUYET'}, f'{rid}: sai review_status')
            if row['review_status'] == 'DA_DUYET':
                check(bool(row['reviewer'].strip() and row['review_note'].strip()), f'{rid}: duyệt thiếu tên/lý do')
        else:
            check(row['review_status'] == 'CHUA_DUYET', f'{rid}: phải CHUA_DUYET ban đầu')
            check(row['reviewer'] == row['review_note'] == '', f'{rid}: thông tin duyệt ban đầu phải trống')
        if row['variant_type'] == 'ASR_LIKE':
            check(bool(row['clean_reference_text'].strip() and row['asr_noise_note'].strip()), f'{rid}: thiếu cặp ASR')
            check(normalize(row['text']) != normalize(row['clean_reference_text']), f'{rid}: ASR không khác câu sạch')
        else:
            check(row['clean_reference_text'] == row['asr_noise_note'] == '', f'{rid}: CLEAN không cần cột ASR')
        for field in ['text', 'clean_reference_text']:
            if not row[field]:
                continue
            normalized = normalize(row[field])
            check(bool(normalized), f'{rid}: text chỉ có dấu câu')
            if normalized in old_texts:
                collisions.append({'id': rid, 'field': field, 'matches': old_texts[normalized]})
            check(normalized not in seen_texts, f'{rid}/{field}: trùng nội dung với {seen_texts.get(normalized)}')
            seen_texts[normalized] = rid + '/' + field
        families[row['family_id']].append(row['proposed_label'])
    check(not collisions, 'Có text/câu sạch ASR trùng dữ liệu cũ')
    check(len(families) == 20, 'Cần đúng 20 family')
    for family, family_labels in families.items():
        check(len(family_labels) == 3, f'{family}: cần 3 dòng')
        if not review_progress:
            check(Counter(family_labels) == Counter({x: 1 for x in LABELS}), f'{family}: phải có 1 câu mỗi nhãn')
    if not review_progress:
        check(labels == Counter({x: 20 for x in LABELS}), 'Phân bố ban đầu phải 20/20/20')
    _, audio = read_csv(root / 'data/audio_test_ground_truth.csv')
    check(len(audio) == 9 and len({r['id'] for r in audio}) == 9, 'Audio phải có 9 ID riêng')
    check(Counter(r['true_label'] for r in audio) == Counter(KHONG_CANH_BAO=4, CAN_XAC_MINH=2, CANH_BAO=3), 'Audio phải 4/2/3')
    check(all(r.get('review_status') == 'DA_DUYET' and r.get('reviewer') == 'Vũ Hồng Phúc' for r in audio), 'Audio thiếu xác nhận duyệt')
    at8 = next((r for r in audio if r['id'] == 'AT008'), {})
    check(at8.get('label_before_review') == 'CAN_XAC_MINH' and at8.get('true_label') == 'KHONG_CANH_BAO', 'AT008 sai dấu vết đổi nhãn')
    audit_path = root / 'reports/data_candidates_v03/preparation_audit.json'
    protected_count = 0
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        for relative, expected in audit['protected_file_sha256_before'].items():
            p = root / relative
            with p.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            check(actual == expected, f'File được bảo vệ đã đổi: {relative}')
            protected_count += 1
        previous = {r['id']: r for r in audit['audio_before_records']}
        for row in audio:
            old_row = previous.get(row['id'], {})
            check(row['text'] == old_row.get('text'), f"{row['id']}: không được sửa câu audio")
            if row['id'] != 'AT008':
                check(row['true_label'] == old_row.get('true_label') and row['reason'] == old_row.get('reason'), f"{row['id']}: thay nhãn/lý do ngoài phạm vi")
    return dict(status='VALID' if not errors else 'INVALID', stage='review_progress' if review_progress else 'initial_candidates',
        errors=errors, rows=len(rows), label_counts=dict(labels), family_count=len(families),
        variant_counts=dict(variants), review_counts=dict(Counter(r['review_status'] for r in rows)),
        unique_ids=len({r['id'] for r in rows}), unique_candidate_texts=len({normalize(r['text']) for r in rows}),
        normalized_text_overlap=collisions, historical_sources=sorted({r['source'] for r in old}),
        historical_unique_texts=len(old_texts), protected_files_verified=protected_count,
        audio_label_counts=dict(Counter(r['true_label'] for r in audio)),
        attention_ids=[r['id'] for r in rows if r['attention_note']],
        asr_pair_ids=[r['id'] for r in rows if r['variant_type'] == 'ASR_LIKE'],
        normalization='NFC + casefold + punctuation as boundaries + collapse whitespace; keep accents and digits',
        semantic_equivalence_verified_automatically=False, training_performed=False, inference_performed=False)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, default=ROOT / 'data/candidates_v03_review.csv')
    parser.add_argument('--review-progress', action='store_true')
    args = parser.parse_args()
    try:
        result = validate(args.candidates.resolve(), args.review_progress)
    except (OSError, ValueError, KeyError) as error:
        result = {'status': 'INVALID', 'errors': [str(error)]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['status'] == 'VALID' else 1)

if __name__ == '__main__':
    main()
