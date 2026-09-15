"""Extract approved Excel and preserve v02 split membership. No model calls."""
import csv
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
import openpyxl

ROOT = Path(__file__).resolve().parent
LABELS = {'KHONG_CANH_BAO', 'CAN_XAC_MINH', 'CANH_BAO'}

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def norm(text):
    return ' '.join(re.findall(r'[^\W_]+', unicodedata.normalize('NFC', text).casefold()))

def texts(rows):
    return {norm(r[k]) for r in rows for k in ['text', 'clean_reference_text'] if r.get(k)}

def write(path, rows, fields):
    with path.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval='')
        writer.writeheader()
        writer.writerows(rows)

def main():
    data = ROOT / 'data'
    outputs = [data/'candidates_v03_approved.csv', data/'dataset_livestream_v03.csv', data/'splits_v03']
    if any(p.exists() for p in outputs):
        raise SystemExit('Outputs already exist; refusing to overwrite.')
    original = {r['id']: r for r in read(data/'candidates_v03_review.csv')}
    workbook = openpyxl.load_workbook(data/'candidates_v03_reviewed.xlsx', read_only=True, data_only=True)
    rows, skipped, headers = [], [], []
    for sheet in workbook:
        values = list(sheet.values)
        nonempty = [(i, r) for i, r in enumerate(values, 1) if any(v not in (None, '') for v in r)]
        if not nonempty:
            skipped.append(sheet.title)
            continue
        found = [(i,r) for i,r in nonempty if {'id','text','proposed_label','family_id','review_status'} <= {str(v).strip() for v in r if v is not None}]
        if len(found) != 1:
            raise ValueError(f'{sheet.title}: cannot identify one real header')
        index, header = found[0]
        header = [str(v).strip() if v is not None else '' for v in header]
        active = [i for i,h in enumerate(header) if h]
        if len({header[i] for i in active}) != len(active):
            raise ValueError(f'{sheet.title}: duplicated column name')
        headers.append({'sheet':sheet.title,'header_row':index,'ignored_prefix_rows':index-1})
        for line, values in nonempty:
            if line <= index:
                continue
            if any(v not in ('',None) for i,v in enumerate(values) if i not in active):
                raise ValueError(f'{sheet.title}:{line}: unnamed column contains data')
            rows.append({header[i]: str(values[i]) if values[i] is not None else '' for i in active})
    workbook.close()
    errors, changes, changed_text, filled, asr_checks = [], [], [], 0, []
    if len(rows)!=60: errors.append(f'ROW_COUNT: {len(rows)}, expected 60')
    counts = Counter(r.get('id','') for r in rows)
    for rid,count in counts.items():
        if not rid or count != 1: errors.append(f'{rid or "MISSING_ID"}: count={count}')
    for rid in sorted(set(original)-set(counts)): errors.append(f'{rid}: missing')
    for row in rows:
        rid=row.get('id','')
        if rid not in original:
            errors.append(f'{rid}: unknown ID'); continue
        old=original[rid]
        for key in ['text','family_id']:
            if not row.get(key,'').strip(): errors.append(f'{rid}: empty {key}')
        for key,expected in [('review_status','DA_DUYET'),('reviewer','Vũ Hồng Phúc'),('source','TARGETED_ERROR_ANALYSIS_V03')]:
            if row.get(key)!=expected: errors.append(f'{rid}: {key}={row.get(key)!r}')
        if row.get('proposed_label') not in LABELS: errors.append(f'{rid}: invalid label {row.get("proposed_label")}')
        if row.get('variant_type') not in {'CLEAN','ASR_LIKE'}: errors.append(f'{rid}: invalid variant_type')
        if row.get('family_id') != old['family_id']: errors.append(f'{rid}: family changed')
        for key,value in row.items():
            if any(token in value for token in ['Ã','Ä','Æ','áº','á»','�']): errors.append(f'{rid}: encoding issue in {key}')
        if row['text'] != old['text']: changed_text.append(rid)
        if row['variant_type']=='ASR_LIKE':
            if not row.get('clean_reference_text'): errors.append(f'{rid}: no clean reference')
            if row['text'] != old['text'] or row.get('clean_reference_text') != old.get('clean_reference_text'):
                errors.append(f'{rid}: ASR pair changed; meaning requires review')
            # "chỉ" in "địa chỉ" / "kim chỉ" is a noun, not a negation.
            important = lambda t: Counter(w for w in norm(t).split() if w in {'không','chưa','chẳng','đừng','vẫn','nhưng'})
            same_negation = important(row['text']) == important(row.get('clean_reference_text',''))
            if not same_negation: errors.append(f'{rid}: negation/contrast differs in ASR pair')
            asr_checks.append({'id':rid,'unchanged_approved_pair':True,'negation_preserved':same_negation,
                               'semantic_review':'Same original near-sound replacement in non-decisive noun; claims/numbers/fraud cues preserved.'})
        row['label_before_review']=old['proposed_label']
        row['label']=row['proposed_label']
        if row['label']!=old['proposed_label']:
            changes.append({'id':rid,'before':old['proposed_label'],'after':row['label']})
        if not row.get('review_note','').strip():
            row['review_note'] = ('Đã đọc và xác nhận giữ nguyên nhãn theo tiêu chí nghiệp vụ.' if row['label']==old['proposed_label']
                else f"Người duyệt xác nhận đổi nhãn từ {old['proposed_label']} sang {row['label']} sau khi đọc lại theo tiêu chí nghiệp vụ.")
            filled+=1
    fc=Counter(r.get('family_id') for r in rows)
    if len(fc)!=20 or set(fc.values())!={3}: errors.append('FAMILY_COUNTS: expected 20 families of 3')
    if errors:
        raise SystemExit('\n'.join(errors))
    old_rows=read(data/'dataset_livestream_v02.csv')
    train_old=read(data/'splits_v02/train.csv'); val_old=read(data/'splits_v02/validation.csv')
    assert len(old_rows)==150 and len(train_old)==120 and len(val_old)==30
    assert not {r['family_id'] for r in rows}&{r['family_id'] for r in old_rows}
    merged=old_rows+rows
    assert len({r['id'] for r in merged})==210
    assert len({norm(r['text']) for r in merged})==210
    heldout=read(data/'final_test_v03_chua_duyet.csv')+read(data/'audio_test_ground_truth.csv')
    assert not {r['id'] for r in merged}&{r['id'] for r in heldout}
    assert not texts(merged)&texts(heldout)
    # Compare saved ASR units/segments; their text is never added to dataset.
    for path in (ROOT/'reports').glob('audio_demo_*/result.json'):
        result=json.loads(path.read_text(encoding='utf-8'))
        for key in ['classification_units','whisper_segments','segments']:
            assert not texts(merged)&texts(result.get(key,[])), f'ASR overlap: {path}/{key}'
    families=sorted(fc)
    random.Random(42).shuffle(families)
    new_train=set(families[:16]); new_val=set(families[16:])
    train_ids={r['id'] for r in train_old}; val_ids={r['id'] for r in val_old}
    train=[dict(r,split='train') for r in merged if r['id'] in train_ids or r['family_id'] in new_train]
    val=[dict(r,split='validation') for r in merged if r['id'] in val_ids or r['family_id'] in new_val]
    assert len(train)==168 and len(val)==42
    for key in ['id','family_id']:
        assert not {r[key] for r in train}&{r[key] for r in val}
    assert not texts(train)&texts(val)
    for split in [train,val]: assert {r['label'] for r in split}==LABELS
    # Old text/label/family and original membership are all kept exactly.
    for old in train_old+val_old:
        new=next(r for r in train+val if r['id']==old['id'])
        assert all(new[k]==old[k] for k in ['text','label','family_id','split'])
    fields=list(dict.fromkeys(k for r in merged for k in r))
    if 'split' not in fields: fields.append('split')
    membership={r['id']:r['split'] for r in train+val}
    merged=[dict(r,split=membership[r['id']]) for r in merged]
    outputs[2].mkdir()
    write(outputs[0],rows,list(rows[0])); write(outputs[1],merged,fields)
    write(outputs[2]/'train.csv',train,fields);write(outputs[2]/'validation.csv',val,fields)
    summary={'valid':True,'excel_headers':headers,'ignored_empty_sheets':skipped,'approved_rows':60,
             'label_changes':changes,'text_changes':changed_text,'review_notes_filled':filled,'asr_checks':asr_checks,
             'seed':42,'new_train_families':sorted(new_train),'new_validation_families':sorted(new_val),
             'split_method':'random.Random(42).shuffle(sorted families), first 16 train; original v02 membership preserved',
             'leakage_checks_passed':True,'clean_reference_used_as_training_row':False,
             'train':{'rows':len(train),'labels':dict(Counter(r['label'] for r in train))},
             'validation':{'rows':len(val),'labels':dict(Counter(r['label'] for r in val))}}
    (outputs[2]/'split_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='asr_checks'},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
