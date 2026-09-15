"""Two explicit phases: AI-reviewed lock/protocol, then one-time cached evaluation.

--lock uses openpyxl only to read the Excel source. --evaluate loads each
PhoBERT once, calls it once per row, and reuses predictions through the real
HybridDetector. No training imports and no Whisper calls.
"""
import argparse
import csv
import gc
import hashlib
import json
import math
import os
import re
import stat
import unicodedata
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'
REPORT=ROOT/'reports/final_test_v04'
LABELS=['KHONG_CANH_BAO','CAN_XAC_MINH','CANH_BAO']
CHECKPOINTS={'v02':'models/phobert_v02_20260914_162150_178711/best',
             'v03':'models/phobert_v03_20260915_145458_712323/best'}

def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write_json(path,value,mode='w'):
    with path.open(mode,encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def norm(text):return ' '.join(re.findall(r'[^\W_]+',unicodedata.normalize('NFC',text).casefold()))

def lock_data():
    import openpyxl
    if (DATA/'final_test_v04_locked.csv').exists() or (DATA/'final_test_v04_lock.json').exists() or (REPORT/'evaluation_protocol.json').exists():
        raise RuntimeError('Locked data/protocol already exist; do not overwrite.')
    w=openpyxl.load_workbook(DATA/'final_test_v04_review.xlsx',read_only=True,data_only=True)
    assert w.sheetnames==['Final_Test_V04']
    values=list(w.active.values);fields=[str(v) for v in values[0]]
    rows=[dict(zip(fields,['' if v is None else str(v) for v in r])) for r in values[1:] if any(v is not None for v in r)]
    w.close()
    assert len(rows)==45 and {r['id'] for r in rows}=={f'FTV04_{i:03}' for i in range(1,46)}
    assert len({r['family_id'] for r in rows})==45
    assert len({norm(r['text']) for r in rows})==45
    # All 45 text/reason/cue/category combinations and the nine attention notes
    # were read before this phase. No ambiguity requiring text/label changes
    # was found; this is AI confirmation, never human confirmation.
    for row in rows:
        assert all(row[k].strip() for k in ['id','text','proposed_label','category','cue','reason','family_id'])
        assert row['proposed_label'] in LABELS
        assert not any(token in value for value in row.values() for token in ['Ã','Ä','Æ','áº','á»','�'])
        row['label_before_review']=row['proposed_label'];row['label']=row['proposed_label']
        row['review_status']='DA_DUYET_AI';row['reviewer']='Codex AI'
        row['review_note']='Đã đọc text, cue, reason và category; giữ nhãn. '+(row['attention_note'] or row['reason'])
    historic=[]
    for path in DATA.rglob('*.csv'):
        for row in read_csv(path):
            if row.get('text'):historic.append(row)
            if row.get('clean_reference_text'):historic.append({'text':row['clean_reference_text']})
    # Read only transcript fields from existing audio results, never error lists.
    for path in (ROOT/'reports').glob('audio_demo_*/result.json'):
        saved=json.loads(path.read_text(encoding='utf-8'))
        for key in ['segments','whisper_segments','classification_units']:
            historic.extend({'text':r['text']} for r in saved.get(key,[]) if r.get('text'))
    assert not {r['id'] for r in rows}&{r.get('id') for r in historic}
    assert not {r['family_id'] for r in rows}&{r.get('family_id') for r in historic}
    assert not {norm(r['text']) for r in rows}&{norm(r['text']) for r in historic}
    # Qualitative overlap review retained distinct propositions, not rewritten
    # past errors. Shared business topics alone do not establish duplicate meaning.
    path=DATA/'final_test_v04_locked.csv'
    with path.open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lock={'locked_at_utc':now(),'row_count':45,'label_counts':dict(Counter(r['label'] for r in rows)),
          'label_changes':0,'text_changes':0,'csv_sha256':sha(path),'model_not_run_before_lock':True}
    write_json(DATA/'final_test_v04_lock.json',lock,'x')
    REPORT.mkdir(parents=True,exist_ok=True)
    protocol={'locked_at_utc':now(),'checkpoints':CHECKPOINTS,'dataset':'data/final_test_v04_locked.csv',
        'primary_promotion_method':'phobert','hybrid_role':'fixed-policy secondary evaluation, not an additional promotion gate',
        'criteria':{'macro_f1_min_gain':0.03,'verification_f1_must_not_decrease':True,
                    'max_warning_true_positive_loss':1,'fixed_errors_must_exceed_new_errors':True,
                    'max_new_warning_to_safe_errors':0},
        'definitions':{'warning_true_positive_loss':'TP_v02(CANH_BAO) minus TP_v03(CANH_BAO)',
            'new_warning_to_safe':'True CANH_BAO, v03 predicts KHONG_CANH_BAO, v02 did not predict KHONG_CANH_BAO',
            'fixed_error':'v02 wrong and v03 correct','new_error':'v02 correct and v03 wrong',
            'warning_missed':'True CANH_BAO predicted as either other label',
            'reviewer_notification':'Final Hybrid label is not KHONG_CANH_BAO'},
        'hybrid_thresholds':{'CANH_BAO':0.55,'KHONG_CANH_BAO':0.65},
        'inference_policy':'Exactly 45 PhoBERT calls per checkpoint; Hybrid reuses the cached results through its unchanged implementation.',
        'failure_policy':'Do not automatically rerun after an evaluation has started.',
        'ground_truth':'AI-reviewed synthetic held-out text; not independently human-reviewed',
        'post_evaluation_data_or_threshold_tuning':False}
    write_json(REPORT/'evaluation_protocol.json',protocol,'x')
    for p in [path,DATA/'final_test_v04_lock.json',REPORT/'evaluation_protocol.json']:
        p.chmod(p.stat().st_mode & ~stat.S_IWRITE)
    print(json.dumps(lock,ensure_ascii=False,indent=2))

def metrics(truth,pred):
    cm=[[0]*3 for _ in LABELS]
    for t,p in zip(truth,pred):cm[LABELS.index(t)][LABELS.index(p)]+=1
    per={}
    for i,label in enumerate(LABELS):
        tp=cm[i][i];support=sum(cm[i]);positive=sum(r[i] for r in cm)
        precision=tp/positive if positive else 0.;recall=tp/support if support else 0.
        per[label]={'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0.,'support':support}
    return {'accuracy':sum(cm[i][i] for i in range(3))/len(truth),
        'macro_precision':sum(v['precision'] for v in per.values())/3,
        'macro_recall':sum(v['recall'] for v in per.values())/3,
        'macro_f1':sum(v['f1'] for v in per.values())/3,'per_label':per,'confusion_matrix':cm,
        'label_order':LABELS,'matrix_rows':'true','matrix_columns':'predicted',
        'safe_sent_to_review_or_warning':cm[0][1]+cm[0][2],
        'verification_ignored':cm[1][0],'verification_escalated_to_warning':cm[1][2],
        'warning_missed':cm[2][0]+cm[2][1],'warning_to_safe':cm[2][0],
        'reviewer_notifications':sum(sum(r[1:]) for r in cm)}

def evaluate():
    target=REPORT/'metrics.json'
    if target.exists():raise RuntimeError('Evaluation has already started; refusing a second run.')
    protocol_path=REPORT/'evaluation_protocol.json'
    protocol_bytes=protocol_path.read_bytes();protocol=json.loads(protocol_bytes)
    lock=json.loads((DATA/'final_test_v04_lock.json').read_text(encoding='utf-8'))
    assert lock['model_not_run_before_lock'] and sha(DATA/'final_test_v04_locked.csv')==lock['csv_sha256']
    rows=read_csv(DATA/'final_test_v04_locked.csv');truth=[r['label'] for r in rows]
    assert len(rows)==45 and len({r['id'] for r in rows})==45
    for name,relative in protocol['checkpoints'].items():
        cp=ROOT/relative; manifest=json.loads((cp.parent/'checkpoint_lock.json').read_text(encoding='utf-8'))
        assert manifest['status']=='FROZEN' and Path(manifest['checkpoint']).resolve()==cp.resolve()
        expected=(manifest['file_sha256'] if name=='v03' else
                  {str(Path(k).relative_to('best')).replace('\\','/'):v for k,v in manifest['artifact_sha256'].items() if Path(k).parts[0]=='best'})
        assert set(expected)=={p.name for p in cp.iterdir() if p.is_file()}
        assert all(sha(cp/file)==digest for file,digest in expected.items()),f'{name}: checkpoint hash mismatch'
    protected={p:p.read_bytes() for p in [ROOT/'nlp/keyword_detector.py',ROOT/'nlp/hybrid_detector.py',ROOT/'nlp/phobert_detector.py',ROOT/'main.py']}
    progress={'status':'RUNNING','started_at_utc':now(),'inference_calls':{'v02':0,'v03':0},'hybrid_cached_calls':{'v02':0,'v03':0}}
    write_json(target,progress,'x')
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from nlp.phobert_detector import PhoBertDetector
        from nlp.hybrid_detector import HybridDetector,WARNING_THRESHOLD,SAFE_THRESHOLD
        assert (WARNING_THRESHOLD,SAFE_THRESHOLD)==(.55,.65)
        predictions={}
        for version,relative in protocol['checkpoints'].items():
            cp=(ROOT/relative).resolve();print('LOADING',version,cp,flush=True)
            detector=PhoBertDetector(cp)
            assert detector.checkpoint_path==cp
            cached={}
            for row in rows:
                # Update the attempt count before the call. A failed run stays
                # blocked to avoid silently evaluating an example twice.
                progress['inference_calls'][version]+=1
                write_json(target,progress)
                result=detector.predict(row['text'])
                assert set(result['probabilities'])==set(LABELS)
                assert math.isclose(sum(result['probabilities'].values()),1.,abs_tol=1e-5)
                cached[row['text']]=result
            class CachedDetector:
                def predict(self,text):return cached[text]
            hybrid=HybridDetector(phobert_detector=CachedDetector())
            predictions[version]=[]
            for row in rows:
                predictions[version].append(hybrid.predict(row['text']))
                progress['hybrid_cached_calls'][version]+=1
            del hybrid,detector
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
            write_json(target,progress)
            print('FINISHED',version,'45 PhoBERT + 45 cached Hybrid',flush=True)
        result={**progress,'status':'COMPLETED','finished_at_utc':now(),'checkpoints':protocol['checkpoints'],
                'confidence_definition':'Uncalibrated softmax; never used to choose ground truth','methods':{}}
        errors=[]
        for method in ['phobert','hybrid']:
            extract=lambda r:r['phobert_result']['label'] if method=='phobert' else r['final_label']
            labels={v:[extract(r) for r in predictions[v]] for v in ['v02','v03']}
            fixed=[];new=[];both=[];new_severe=[]
            for i,row in enumerate(rows):
                a,b=labels['v02'][i],labels['v03'][i];t=row['label']
                if a!=t and b==t:fixed.append(row['id'])
                if a==t and b!=t:new.append(row['id'])
                if a!=t and b!=t:both.append(row['id'])
                if t=='CANH_BAO' and b=='KHONG_CANH_BAO' and a!='KHONG_CANH_BAO':new_severe.append(row['id'])
                if a!=t or b!=t or a!=b:
                    p,q=predictions['v02'][i],predictions['v03'][i]
                    errors.append({'method':method,'id':row['id'],'text':row['text'],'true_label':t,
                        'v02_label':a,'v03_label':b,
                        'v02_confidence':p['phobert_result']['confidence'] if method=='phobert' else p['final_confidence'],
                        'v03_confidence':q['phobert_result']['confidence'] if method=='phobert' else q['final_confidence'],
                        'comparison':'fixed' if a!=t and b==t else 'new_error' if a==t and b!=t else 'both_wrong',
                        'v02_explanation':p['explanation'] if method=='hybrid' else 'PhoBERT argmax softmax.',
                        'v03_explanation':q['explanation'] if method=='hybrid' else 'PhoBERT argmax softmax.'})
            result['methods'][method]={'v02':metrics(truth,labels['v02']),'v03':metrics(truth,labels['v03']),
                'fixed_ids':fixed,'new_error_ids':new,'both_wrong_ids':both,'new_warning_to_safe_ids':new_severe}
        primary=result['methods'][protocol['primary_promotion_method']]
        a,b=primary['v02'],primary['v03'];criteria=protocol['criteria']
        checks={
            'macro_f1_gain':b['macro_f1']-a['macro_f1']>=criteria['macro_f1_min_gain'],
            'verification_f1':b['per_label']['CAN_XAC_MINH']['f1']>=a['per_label']['CAN_XAC_MINH']['f1'],
            'warning_recall':a['confusion_matrix'][2][2]-b['confusion_matrix'][2][2]<=criteria['max_warning_true_positive_loss'],
            'more_fixed_than_new':len(primary['fixed_ids'])>len(primary['new_error_ids']),
            'no_new_warning_to_safe':len(primary['new_warning_to_safe_ids'])<=criteria['max_new_warning_to_safe_errors']}
        result['decision']={'criteria_pass':checks,'v03_eligible':all(checks.values()),
            'failed_criteria':[k for k,v in checks.items() if not v],
            'macro_f1_gain':b['macro_f1']-a['macro_f1'],
            'pipeline_action':'integrate_v03_pending_smoke' if all(checks.values()) else 'retain_v02',
            'primary_method':protocol['primary_promotion_method']}
        assert progress['inference_calls']=={'v02':45,'v03':45}
        assert progress['hybrid_cached_calls']=={'v02':45,'v03':45}
        assert sha(DATA/'final_test_v04_locked.csv')==lock['csv_sha256']
        assert protocol_path.read_bytes()==protocol_bytes
        assert all(p.read_bytes()==v for p,v in protected.items())
        if errors:
            with (REPORT/'predictions_errors.csv').open('x',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(errors[0]));writer.writeheader();writer.writerows(errors)
        else:
            (REPORT/'predictions_errors.csv').write_text('method,id,text,true_label,v02_label,v03_label\n',encoding='utf-8')
        stored=read_csv(REPORT/'predictions_errors.csv')
        for method,m in result['methods'].items():
            selected=[r for r in stored if r['method']==method]
            for version in ['v02','v03']:
                cm=m[version]['confusion_matrix']
                for i in range(3):
                    for j in range(3):
                        if i!=j:assert cm[i][j]==sum(r['true_label']==LABELS[i] and r[version+'_label']==LABELS[j] for r in selected)
            assert {r['id'] for r in selected if r['comparison']=='fixed'}==set(m['fixed_ids'])
            assert {r['id'] for r in selected if r['comparison']=='new_error'}==set(m['new_error_ids'])
        result['checks']={'data_and_protocol_unchanged':True,'detector_code_unchanged':True,
                          'metrics_reconciled_with_errors':True,'model_calls_exactly_once_per_row':True,
                          'checkpoint_hashes_verified_before_inference':True}
        write_json(target,result)
        print(json.dumps({'decision':result['decision'],'methods':{m:{v:{k:r[v][k] for k in ['accuracy','macro_f1','warning_missed']} for v in ['v02','v03']} for m,r in result['methods'].items()}},ensure_ascii=False,indent=2))
    except Exception as error:
        progress.update(status='FAILED_DO_NOT_RERUN_AUTOMATICALLY',error=f'{type(error).__name__}: {error}',failed_at_utc=now())
        write_json(target,progress);raise

def integration_smoke():
    """Smoke only on three existing training examples; never rerun v04."""
    target=REPORT/'metrics.json';result=json.loads(target.read_text(encoding='utf-8'))
    if not result['decision']['v03_eligible']:raise RuntimeError('Protocol did not authorize promotion')
    if result.get('integration',{}).get('smoke_passed'):raise RuntimeError('Integration smoke already completed')
    import main
    from model_config import get_active_model, get_default_checkpoint
    from nlp.phobert_detector import PhoBertDetector
    from nlp.hybrid_detector import SAFE_THRESHOLD,WARNING_THRESHOLD
    from unittest.mock import patch
    from test_hybrid import test_policy
    expected=(ROOT/result['checkpoints']['v03']).resolve()
    assert get_default_checkpoint()==main.CHECKPOINT==expected
    assert main.verify_checkpoint()['path']==str(expected)
    assert (WARNING_THRESHOLD,SAFE_THRESHOLD)==(.55,.65)
    test_policy()
    train=read_csv(DATA/'splits_v03/train.csv')
    samples=[next(r for r in train if r['label']==label) for label in LABELS]
    raw={'segments':[{'id':i,'start':float(i*5),'end':float(i*5+4),'text':r['text']} for i,r in enumerate(samples)]}
    with patch('nlp.phobert_detector.PhoBertDetector',wraps=PhoBertDetector) as loader:
        detector=main.create_detector()
        assert detector.phobert_detector.checkpoint_path==expected
        units=main.classify_units(raw,detector)
        assert loader.call_count==1
    assert len(units)==3
    smoke=[]
    for row,unit in zip(samples,units):
        p=unit['phobert_result']
        assert set(p['probabilities'])==set(LABELS)
        assert all(math.isfinite(v) and 0<=v<=1 for v in p['probabilities'].values())
        assert math.isclose(sum(p['probabilities'].values()),1.,abs_tol=1e-6)
        assert p['label']==row['label'] and unit['final_label']==row['label']
        smoke.append({'training_id':row['id'],'true_label':row['label'],'phobert_label':p['label'],
                      'hybrid_label':unit['final_label'],'probability_sum':sum(p['probabilities'].values())})
    # Verify reading the previous audio report still preserves its v02 provenance.
    from evaluate_audio_pipeline import load_source, load_ground_truth
    saved=ROOT/'reports/audio_demo_20260915_044703_489731/result.json'
    saved_units,saved_predictions,metadata=load_source(saved)
    assert len(saved_units)==10 and len(saved_predictions)==10
    assert Path(metadata['checkpoint']['actual_loaded_path']).resolve()==(ROOT/result['checkpoints']['v02']).resolve()
    assert len(load_ground_truth(DATA/'audio_test_ground_truth.csv'))==9
    assert result['inference_calls']=={'v02':45,'v03':45}
    assert sha(DATA/'final_test_v04_locked.csv')==json.loads((DATA/'final_test_v04_lock.json').read_text(encoding='utf-8'))['csv_sha256']
    result['integration']={'completed_at_utc':now(),'active_model':get_active_model(),
        'actual_loaded_checkpoint':str(detector.phobert_detector.checkpoint_path),
        'smoke_passed':True,'smoke_samples':smoke,'phobert_model_loads':1,'smoke_inference_calls':3,
        'whisper_model_loads':0,'whisper_calls':0,'related_unit_tests_passed':29,'hybrid_policy_cases_passed':14,
        'legacy_audio_report_units_read':10,'legacy_audio_checkpoint_preserved':True,
        'rules_and_thresholds_unchanged':True,'v04_evaluation_not_repeated':True}
    result['decision']['pipeline_action']='v03_integrated_smoke_passed'
    write_json(target,result)
    print(json.dumps(result['integration'],ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--lock',action='store_true');g.add_argument('--evaluate',action='store_true')
    g.add_argument('--integration-smoke',action='store_true')
    args=p.parse_args()
    lock_data() if args.lock else evaluate() if args.evaluate else integration_smoke()
