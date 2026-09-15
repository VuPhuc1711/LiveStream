"""Compare frozen v03 with v02; stored audio only, no training or Whisper."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import argparse
import csv
import gc
import hashlib
import json
import math
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from train_phobert_v03 import LABELS, LABEL2ID, ROOT, metrics, write_json

V02=ROOT/'models/phobert_v02_20260914_162150_178711/best'

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def verify_checkpoint(lock):
    checkpoint=Path(lock['checkpoint'])
    assert lock['status']=='FROZEN'
    for name,expected in lock['file_sha256'].items():
        with (checkpoint/name).open('rb') as f:
            assert hashlib.file_digest(f,'sha256').hexdigest()==expected,f'Checkpoint changed: {name}'

def build_benchmarks():
    benches={
        'validation_v03':read(ROOT/'data/splits_v03/validation.csv'),
        'validation_v02':read(ROOT/'data/splits_v02/validation.csv'),
        'development_regression_v03':read(ROOT/'data/final_test_v03_chua_duyet.csv')}
    audio=read(ROOT/'data/audio_test_ground_truth.csv')
    assert len(audio)==9 and next(r for r in audio if r['id']=='AT008')['true_label']=='KHONG_CANH_BAO'
    for row in audio:row['label']=row['true_label']
    candidates=[]
    for path in sorted((ROOT/'reports').glob('audio_demo_*/result.json')):
        data=json.loads(path.read_text(encoding='utf-8'))
        if Path(data.get('audio_path','')).name=='audio_test_9_sentences.m4a' and data.get('classification_units'):
            candidates.append((path,data))
    if not candidates:raise ValueError('No saved sentence-unit report for requested audio')
    audio_path,saved=candidates[-1]
    alignments=[]
    for path in sorted((ROOT/'reports').glob('audio_pipeline_eval_*/report.json')):
        alignment=json.loads(path.read_text(encoding='utf-8'))
        if Path(alignment.get('source_path','')).resolve()==audio_path.resolve(): alignments.append((path,alignment))
    if not alignments:raise ValueError('No saved alignment for latest audio')
    alignment_path,alignment=alignments[-1]
    assert not alignment['alignment_summary']['missing_reference_ids']
    assert not alignment['alignment_summary']['shared_unit_ids']
    owners={}
    for row in alignment['sentences']:
        for uid in row['unit_ids']:
            assert uid not in owners
            owners[uid]=row['id']
    lookup={r['id']:r for r in audio}
    units=[{'id':u['unit_id'],'text':u['text'],'label':lookup[owners[u['unit_id']]]['label'],
            'reference_id':owners[u['unit_id']],'source_segment_id':u['source_segment_id']} for u in saved['classification_units']]
    assert len(units)==10 and set(owners.values())==set(lookup)
    reconstructed=[]
    for row in audio:
        parts=[u['text'] for u in units if u['reference_id']==row['id']]
        assert parts
        reconstructed.append(dict(row,text=' '.join(parts)))
    benches.update(audio_clean=audio,audio_reconstructed_9=reconstructed,audio_classification_units=units)
    provenance={'saved_audio_result':str(audio_path),'saved_alignment':str(alignment_path),
        'audio_file':saved['audio_path'],'whisper_calls':0,'expected_sentences':9,'classification_units':len(units),
        'notes':'Unit labels inherit parent sentence truth. AT005 second unit is context-dependent; reconstructed 9-sentence score is diagnostic, not production pipeline score.'}
    return benches,provenance

def paired_report(rows,old,new,method):
    key='label' if method=='phobert' else 'final_label'
    truth=[LABEL2ID[r['label']] for r in rows]
    a=[LABEL2ID[x['phobert_result'][key] if method=='phobert' else x[key]] for x in old]
    b=[LABEL2ID[x['phobert_result'][key] if method=='phobert' else x[key]] for x in new]
    joint=Counter(zip(truth,a,b))
    # Joint counts suffice to independently recompute both confusion matrices,
    # fixed/new error counts without saving all correct predictions again.
    result={'v02':metrics(truth,a),'v03':metrics(truth,b),
            'fixed_ids':[r['id'] for r,t,p,q in zip(rows,truth,a,b) if p!=t and q==t],
            'new_error_ids':[r['id'] for r,t,p,q in zip(rows,truth,a,b) if p==t and q!=t],
            'paired_counts':[{'truth':t,'v02':p,'v03':q,'count':n} for (t,p,q),n in sorted(joint.items())]}
    # Reconcile persisted sufficient statistics with calculated metrics.
    expanded=[(d['truth'],d['v02'],d['v03']) for d in result['paired_counts'] for _ in range(d['count'])]
    for name,index in [('v02',1),('v03',2)]:
        check=metrics([x[0] for x in expanded],[x[index] for x in expanded])
        assert check['confusion_matrix']==result[name]['confusion_matrix']
        assert math.isclose(check['macro_f1'],result[name]['macro_f1'],abs_tol=1e-12)
    return result

def assess(summary):
    benches=summary['benchmarks']
    validation=benches['validation_v03']['phobert']
    regression=benches['development_regression_v03']['phobert']
    clean=benches['audio_clean']['phobert']
    audio=benches['audio_classification_units']['hybrid']
    official=next(e for e in summary['experiments'] if e['name']==summary['selection']['selected_experiment'])
    repeats=[e for e in summary['experiments'] if e['configuration']['seed']==123]
    return {
        'decision':'Promising validation improvement, not an unqualified replacement for v02; retain both checkpoints and keep pipeline on v02.',
        'validation_macro_f1_delta':validation['v03']['macro_f1']-validation['v02']['macro_f1'],
        'validation_verification_f1_delta':validation['v03']['per_label']['CAN_XAC_MINH']['f1-score']-validation['v02']['per_label']['CAN_XAC_MINH']['f1-score'],
        'validation_warning_recall_delta':validation['v03']['per_label']['CANH_BAO']['recall']-validation['v02']['per_label']['CANH_BAO']['recall'],
        'old_validation_macro_f1_delta':benches['validation_v02']['phobert']['v03']['macro_f1']-benches['validation_v02']['phobert']['v02']['macro_f1'],
        'regression_fixed_new':[len(regression['fixed_ids']),len(regression['new_error_ids'])],
        'clean_audio_macro_f1_delta':clean['v03']['macro_f1']-clean['v02']['macro_f1'],
        'audio_unit_hybrid_fixed_new':[len(audio['fixed_ids']),len(audio['new_error_ids'])],
        'seed123_macro_f1_delta':repeats[0]['best']['macro_f1']-official['best']['macro_f1'] if repeats else None,
        'limitations':['Small synthetic, previously examined benchmarks; no independent final evaluation.',
            'AT002 remains incorrect on both clean and ASR text; AT005 clean remains incorrect.',
            'AT008 clean PhoBERT and ASR Hybrid regress; unchanged thresholds can turn safe argmax into verification.',
            'Seed-123 stability repeat is materially weaker; two seeds do not estimate reliable variance.',
            'Per-unit audio truth is inherited from parent sentences; AT005 evidence-only unit depends on context.'],
        'next_technical_step':'Investigate sentence context and probability calibration on separate development data, then evaluate on a new untouched human-reviewed test. No additional training or threshold changes performed here.'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,required=True)
    args=parser.parse_args()
    summary=json.loads(args.summary.read_text(encoding='utf-8'))
    if summary['status']!='CHECKPOINT_FROZEN_AWAITING_COMPARISON':raise ValueError('Comparison already run or checkpoint not frozen')
    lock=summary['selection'];verify_checkpoint(lock)
    started=datetime.now(timezone.utc).isoformat()
    assert lock['frozen_at_utc']<started
    benches,provenance=build_benchmarks()
    # Snapshot exact bytes of files we must not change, no additional hashes.
    protected=[ROOT/'main.py',ROOT/'nlp/keyword_detector.py',ROOT/'nlp/hybrid_detector.py',
               ROOT/'nlp/phobert_detector.py',ROOT/'final_evaluation_lock.json',
               ROOT/'data/final_test_v03_chua_duyet.csv',ROOT/'data/audio_test_ground_truth.csv']
    before={p:p.read_bytes() for p in protected}
    import torch
    import torch.nn.functional as F
    numeric_logits=torch.tensor([[.2,1.1,-.3],[1.2,-.1,.4]],requires_grad=True)
    numeric_labels=torch.tensor([0,1]);numeric_weights=torch.tensor([.8,1.3,.9])
    full=F.cross_entropy(numeric_logits,numeric_labels,weight=numeric_weights)
    accumulated=sum(F.cross_entropy(numeric_logits[i:i+1],numeric_labels[i:i+1],weight=numeric_weights,reduction='sum')
                    for i in range(2))/numeric_weights[numeric_labels].sum()
    full_grad=torch.autograd.grad(full,numeric_logits,retain_graph=True)[0]
    accumulated_grad=torch.autograd.grad(accumulated,numeric_logits)[0]
    assert torch.allclose(full,accumulated) and torch.allclose(full_grad,accumulated_grad)
    from nlp.phobert_detector import PhoBertDetector
    from nlp.hybrid_detector import HybridDetector,WARNING_THRESHOLD,SAFE_THRESHOLD
    assert WARNING_THRESHOLD==.55 and SAFE_THRESHOLD==.65
    predictions={}
    for name,path in [('v02',V02),('v03',Path(lock['checkpoint']))]:
        print('COMPARE',name,path,flush=True)
        pho=PhoBertDetector(path);hybrid=HybridDetector(phobert_detector=pho)
        cache={}; predictions[name]={}
        for bench,rows in benches.items():
            predictions[name][bench]=[]
            for row in rows:
                if row['text'] not in cache:cache[row['text']]=hybrid.predict(row['text'])
                predictions[name][bench].append(cache[row['text']])
            print('DONE',name,bench,len(rows),flush=True)
        del hybrid,pho;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    comparison={};errors=[]
    for bench,rows in benches.items():
        comparison[bench]={}
        old=predictions['v02'][bench];new=predictions['v03'][bench]
        for method in ['phobert','hybrid']:
            comparison[bench][method]=paired_report(rows,old,new,method)
            for row,p,q in zip(rows,old,new):
                def extract(result):
                    if method=='phobert':return result['phobert_result']['label'],result['phobert_result']['confidence']
                    return result['final_label'],result['final_confidence']
                pl,pc=extract(p);ql,qc=extract(q)
                if pl!=row['label'] or ql!=row['label'] or pl!=ql:
                    errors.append({'benchmark':bench,'method':method,'id':row['id'],'text':row['text'],
                        'true_label':row['label'],'v02_label':pl,'v02_confidence':pc,'v03_label':ql,'v03_confidence':qc,
                        'change':'fixed' if pl!=row['label'] and ql==row['label'] else 'new_error' if pl==row['label'] and ql!=row['label'] else 'persistent_error',
                        'v02_explanation':p['explanation'] if method=='hybrid' else 'PhoBERT argmax; confidence is uncalibrated softmax.',
                        'v03_explanation':q['explanation'] if method=='hybrid' else 'PhoBERT argmax; confidence is uncalibrated softmax.'})
    audio_details=[];strict={name:{m:[] for m in ['phobert','hybrid']} for name in ['v02','v03']}
    for i,row in enumerate(benches['audio_clean']):
        indices=[j for j,u in enumerate(benches['audio_classification_units']) if u['reference_id']==row['id']]
        detail={'id':row['id'],'true_label':row['label'],'unit_ids':[benches['audio_classification_units'][j]['id'] for j in indices]}
        for name in ['v02','v03']:
            detail[name]={}
            for method in ['phobert','hybrid']:
                label=lambda p:p['phobert_result']['label'] if method=='phobert' else p['final_label']
                clean=label(predictions[name]['audio_clean'][i]); asr=label(predictions[name]['audio_reconstructed_9'][i])
                unit_labels=[label(predictions[name]['audio_classification_units'][j]) for j in indices]
                correct=all(x==row['label'] for x in unit_labels)
                strict[name][method].append(correct)
                detail[name][method]={'clean_label':clean,'reconstructed_asr_label':asr,'unit_labels':unit_labels,
                    'all_units_correct':correct,'clean_classification_error':clean!=row['label'],
                    'error_appears_after_asr':clean==row['label'] and asr!=row['label'],
                    'split_context_error':len(indices)>1 and asr==row['label'] and not correct}
        audio_details.append(detail)
    strict_scores={name:{method:{'correct':sum(v),'total':9,'accuracy':sum(v)/9,
        'note':'All units must match the parent sentence label. No invented aggregate label or confusion matrix for a mixed-label sentence.'}
        for method,v in methods.items()} for name,methods in strict.items()}
    fields=['benchmark','method','id','text','true_label','v02_label','v02_confidence','v03_label','v03_confidence','change','v02_explanation','v03_explanation']
    errors_path=args.summary.parent/'predictions_errors.csv'
    with errors_path.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(errors)
    stored=read(errors_path)
    assert len(stored)==len(errors)
    for bench,methods in comparison.items():
        for method,result in methods.items():
            selected=[r for r in stored if r['benchmark']==bench and r['method']==method]
            assert {r['id'] for r in selected if r['change']=='fixed'}==set(result['fixed_ids'])
            assert {r['id'] for r in selected if r['change']=='new_error'}==set(result['new_error_ids'])
            for version in ['v02','v03']:
                cm=result[version]['confusion_matrix']
                assert sum(cm[i][j] for i in range(3) for j in range(3) if i!=j)==sum(r[version+'_label']!=r['true_label'] for r in selected)
    verify_checkpoint(lock)
    assert all(p.read_bytes()==content for p,content in before.items())
    summary.update(status='COMPARISON_COMPLETED',comparison_started_at_utc=started,
        comparison_finished_at_utc=datetime.now(timezone.utc).isoformat(),benchmarks=comparison,
        audio_provenance=provenance,audio_diagnostics=audio_details,audio_strict_9=strict_scores,
        benchmark_roles={'validation_v03':'Model selection set, NOT independent final test',
            'validation_v02':'Old validation subset, also in v03 validation',
            'development_regression_v03':'Previously examined and informed data design; development regression benchmark, NOT independent final test',
            'audio':'Previously examined; regression and ASR diagnostic only'},
        hybrid_thresholds={'warning':WARNING_THRESHOLD,'safe':SAFE_THRESHOLD},
        protected_files_unchanged=True,metrics_reconciled_with_paired_counts_and_errors_csv=True,
        weighted_accumulation_loss_and_gradient_check_passed=True,
        pipeline_checkpoint_replaced=False)
    summary['assessment']=assess(summary)
    write_json(args.summary,summary)
    print('COMPLETED',args.summary,flush=True)
    for bench,methods in comparison.items():
        print(bench,{m:{'v02_f1':v['v02']['macro_f1'],'v03_f1':v['v03']['macro_f1'],
                          'fixed':len(v['fixed_ids']),'new':len(v['new_error_ids'])} for m,v in methods.items()},flush=True)

if __name__=='__main__':main()
