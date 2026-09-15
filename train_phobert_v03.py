"""Controlled E1/E2/E3 (+ one stability repeat) from pretrained PhoBERT only.

No held-out data are read here. Prepare splits separately before execution.
The official seed-42 selection is locked before regression comparisons.
"""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
import gc
import importlib.metadata
import json
import math
import random
import shutil
import stat
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

from train_phobert_v02 import (LABELS, LABEL2ID, MODEL_NAME, BASE_REVISION,
                              normalize_text, read_rows, check_separation, write_json)

ROOT=Path(__file__).resolve().parent
PREPROCESSING={'normalization':'NFC and collapse whitespace; preserve case and punctuation',
               'segmenter':"underthesea.word_tokenize(format='text')",'max_length':128}

def metrics(truth, prediction):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    report=classification_report(truth,prediction,labels=[0,1,2],target_names=LABELS,zero_division=0,output_dict=True)
    return {'accuracy':float(accuracy_score(truth,prediction)),
            'macro_precision':report['macro avg']['precision'],'macro_recall':report['macro avg']['recall'],
            'macro_f1':report['macro avg']['f1-score'], 'per_label':{k:report[k] for k in LABELS},
            'confusion_matrix':confusion_matrix(truth,prediction,labels=[0,1,2]).tolist(),
            'label_order':LABELS,'matrix_rows':'true','matrix_columns':'predicted'}

def run_experiment(run_dir, name, seed, loss_type, lr, epochs, patience, data, tokenizer, encoded):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True, warn_only=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    counts=Counter(r['label'] for r in data['train'])
    inverse=[1/math.sqrt(counts[label]) for label in LABELS]
    weights=[w/(sum(inverse)/3) for w in inverse] if loss_type=='weighted_ce' else [1.,1.,1.]
    weight_tensor=torch.tensor(weights,device=device,dtype=torch.float32)
    config={'experiment':name,'seed':seed,'loss':loss_type,'class_weights':dict(zip(LABELS,weights)),
            'class_weight_source':'train.csv only','class_weight_formula':'inverse sqrt(count), normalized to mean 1',
            'train_label_counts':dict(counts),'learning_rate':lr,'batch_size':1,'gradient_accumulation':8,
            'effective_batch_size':8,'max_epochs':epochs,'early_stopping_patience':patience,'max_length':128,
            'model':MODEL_NAME,'base_revision':BASE_REVISION,'initialized_from_finetuned_checkpoint':False,
            'optimizer':'AdamW','weight_decay':0.01,'gradient_clip_norm':1.0,'scheduler':None,
            'gradient_checkpointing':True,'mixed_precision':'fp16' if device=='cuda' else 'none',
            'tokenizer_use_fast':False,'device':device,'deterministic_algorithms':'warn_only',
            'validation_loss':'unweighted mean CE for comparable tie breaks; weighted loss also recorded',
            'weighted_loss_reduction':'sum(weight[y]*CE)/sum(weight[y]) across each full accumulation group',
            'selection':'highest macro-F1, then lowest unweighted validation loss',
            'early_stopping':'strict macro-F1 improvement resets patience; equal F1 does not',
            'versions':{p:importlib.metadata.version(p) for p in ['torch','transformers','underthesea','numpy','scikit-learn']}}
    directory=run_dir/name;directory.mkdir()
    best=directory/'best'
    collator=DataCollatorWithPadding(tokenizer=tokenizer,return_tensors='pt')
    generator=torch.Generator().manual_seed(seed)
    loaders={n:DataLoader(v,batch_size=1,shuffle=n=='train',collate_fn=collator,num_workers=0,
                          generator=generator if n=='train' else None) for n,v in encoded.items()}
    print(f'BEGIN {name} seed={seed} loss={loss_type} lr={lr} device={device}',flush=True)
    model,info=AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,revision=BASE_REVISION,
        local_files_only=True,use_safetensors=False,num_labels=3,id2label=dict(enumerate(LABELS)),
        label2id=LABEL2ID,problem_type='single_label_classification',attn_implementation='eager',output_loading_info=True)
    if info.get('error_msgs') or info.get('mismatched_keys') or any(not k.startswith('classifier.') for k in info.get('missing_keys',[])):
        raise RuntimeError(f'Base encoder loading failed: {info}')
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.01,foreach=False)
    scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda')
    history=[];best_key=(-1.,-float('inf'));best_f1=-1.;stale=0
    for epoch in range(1,epochs+1):
        model.train();iterator=iter(loaders['train']);train_sum=0.;train_denom=0.;updates=0
        while True:
            group=list(islice(iterator,8))
            if not group: break
            optimizer.zero_grad(set_to_none=True)
            # Crucial for batch size one: do NOT reduce weighted CE separately
            # per microbatch, where the only sample's weight would cancel.
            denom=sum(weights[int(b['labels'].item())] for b in group)
            for batch in group:
                labels=batch.pop('labels').to(device)
                batch={k:v.to(device) for k,v in batch.items()}
                with torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):
                    logits=model(**batch).logits
                weighted=F.cross_entropy(logits.float(),labels,weight=weight_tensor,reduction='sum')
                if not torch.isfinite(weighted): raise RuntimeError('Non-finite training loss')
                train_sum+=weighted.item();train_denom+=weights[int(labels.item())]
                scaler.scale(weighted/denom).backward()
                del logits,weighted,batch,labels
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optimizer);scaler.update();updates+=1
            if updates%7==0: print(f'{name} epoch={epoch} update={updates}/21',flush=True)
        optimizer.zero_grad(set_to_none=True)
        model.eval();truth=[];pred=[];val_sum=0.;weighted_sum=0.;weight_sum=0.
        with torch.inference_mode():
            for batch in loaders['validation']:
                labels=batch.pop('labels').to(device); batch={k:v.to(device) for k,v in batch.items()}
                with torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):
                    logits=model(**batch).logits.float()
                ce=F.cross_entropy(logits,labels,reduction='sum')
                if not torch.isfinite(ce): raise RuntimeError('Non-finite validation loss')
                val_sum+=ce.item();weighted_sum+=ce.item()*weights[int(labels.item())];weight_sum+=weights[int(labels.item())]
                truth.extend(labels.cpu().tolist());pred.extend(logits.argmax(-1).cpu().tolist())
        record={'epoch':epoch,'train_loss':train_sum/train_denom,'validation_loss':val_sum/len(truth),
                'weighted_validation_loss':weighted_sum/weight_sum,**metrics(truth,pred)}
        history.append(record);write_json(directory/'history.json',history)
        key=(record['macro_f1'],-record['validation_loss'])
        if key>best_key:
            best_key=key
            model.save_pretrained(best);tokenizer.save_pretrained(best)
            write_json(best/'preprocessing.json',PREPROCESSING)
            write_json(best/'training_config.json',config)
            # Compact label vectors permit independent metric reconciliation;
            # no full-text list of correctly predicted examples is duplicated.
            saved={**record,'validation_ids':[r['id'] for r in data['validation']],
                   'true_label_ids':truth,'predicted_label_ids':pred}
            write_json(best/'validation_metrics.json',saved)
        if record['macro_f1']>best_f1:
            best_f1=record['macro_f1'];stale=0
        else: stale+=1
        print(f"EPOCH {name} {epoch}: loss={record['validation_loss']:.6f} accuracy={record['accuracy']:.6f} macro_F1={record['macro_f1']:.6f}",flush=True)
        if patience and stale>=patience: break
    del model,optimizer,scaler,loaders
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    result=json.loads((best/'validation_metrics.json').read_text(encoding='utf-8'))
    assert metrics(result['true_label_ids'],result['predicted_label_ids'])['confusion_matrix']==result['confusion_matrix']
    assert math.isclose(metrics(result['true_label_ids'],result['predicted_label_ids'])['macro_f1'],result['macro_f1'],abs_tol=1e-12)
    return {'name':name,'checkpoint':str(best),'configuration':config,
            'best':{k:v for k,v in result.items() if k not in ['validation_ids','true_label_ids','predicted_label_ids']}}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    import torch
    from transformers import AutoTokenizer
    from underthesea import word_tokenize
    data={n:read_rows(ROOT/'data/splits_v03'/f'{n}.csv') for n in ['train','validation']}
    check_separation(data['train'],data['validation'])
    assert len(data['train'])==168 and len(data['validation'])==42
    original_bytes={n:(ROOT/'data/splits_v03'/f'{n}.csv').read_bytes() for n in data}
    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME,revision=BASE_REVISION,use_fast=False,local_files_only=True)
    encoded={};truncated={}
    for name,rows in data.items():
        samples=[];truncated[name]=[]
        for row in rows:
            segmented=word_tokenize(normalize_text(row['text']),format='text')
            if len(tokenizer(segmented,return_token_type_ids=False)['input_ids'])>128: truncated[name].append(row['id'])
            sample=tokenizer(segmented,truncation=True,max_length=128,return_token_type_ids=False)
            sample['labels']=LABEL2ID[row['label']];samples.append(sample)
        encoded[name]=samples
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_dir=(args.output_dir or ROOT/'models'/f'phobert_v03_experiments_{stamp}').resolve()
    run_dir.mkdir(parents=True,exist_ok=False)
    log=(run_dir/'training.log').open('x',encoding='utf-8',buffering=1)
    class Tee:
        def __init__(self,stream): self.stream=stream
        def write(self,text):
            log.write(text);return self.stream.write(text)
        def flush(self): log.flush();self.stream.flush()
        def __getattr__(self,name): return getattr(self.stream,name)
    sys.stdout=Tee(sys.stdout);sys.stderr=Tee(sys.stderr)
    print('EXPERIMENT_DIRECTORY',run_dir,flush=True)
    print('GPU',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',flush=True)
    e1=run_experiment(run_dir,'E1',42,'ce',2e-5,5,None,data,tokenizer,encoded)
    e2=run_experiment(run_dir,'E2',42,'weighted_ce',2e-5,5,None,data,tokenizer,encoded)
    e3_loss='weighted_ce' if e2['best']['macro_f1']-e1['best']['macro_f1']>=.01 else 'ce'
    e3=run_experiment(run_dir,'E3',42,e3_loss,1e-5,8,2,data,tokenizer,encoded)
    experiments=[e1,e2,e3]
    ranked=sorted(experiments,key=lambda e:(e['best']['macro_f1'],-e['best']['validation_loss']),reverse=True)
    selected=ranked[0];gap=ranked[0]['best']['macro_f1']-ranked[1]['best']['macro_f1']
    safety=(ranked[1]['best']['per_label']['CANH_BAO']['recall']-ranked[0]['best']['per_label']['CANH_BAO']['recall']>.05
            and gap<.02)
    if safety: selected=ranked[1]
    selection_reason='Validation macro-F1, tie lower CE loss.' if not safety else 'Macro-F1 gap <0.02 and runner-up warning recall >5 percentage points higher: safety preference.'
    if gap<.02:
        c=selected['configuration']
        stability=run_experiment(run_dir,selected['name']+'_seed123',123,c['loss'],c['learning_rate'],c['max_epochs'],c['early_stopping_patience'],data,tokenizer,encoded)
        experiments.append(stability)
    for n,content in original_bytes.items():
        assert (ROOT/'data/splits_v03'/f'{n}.csv').read_bytes()==content,'Data changed between experiments'
    official=ROOT/'models'/f'phobert_v03_{stamp}'/'best'
    shutil.copytree(selected['checkpoint'],official)
    # Smoke tests use one existing TRAIN row per label, never invent new data.
    from nlp.phobert_detector import PhoBertDetector
    detector=PhoBertDetector(official)
    smoke=[]
    for label in LABELS:
        row=next(r for r in data['train'] if r['label']==label)
        result=detector.predict(row['text'])
        assert set(result['probabilities'])==set(LABELS)
        assert all(0<=p<=1 for p in result['probabilities'].values())
        assert math.isclose(sum(result['probabilities'].values()),1.,abs_tol=1e-5)
        smoke.append({'id':row['id'],'true_label':label,**result})
    del detector;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    import hashlib
    hashes={}
    for path in sorted(official.iterdir()):
        if path.is_file():
            with path.open('rb') as f: hashes[path.name]=hashlib.file_digest(f,'sha256').hexdigest()
            path.chmod(path.stat().st_mode & ~stat.S_IWRITE)
    lock={'status':'FROZEN','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
          'checkpoint':str(official),'file_sha256':hashes,'selected_experiment':selected['name'],
          'selection_reason':selection_reason,'benchmark_used_for_selection':False,
          'stability_seed_policy':'Seed 123 is diagnostic only, not used to replace the seed-42 selection.'}
    write_json(official.parent/'checkpoint_lock.json',lock)
    (official.parent/'checkpoint_lock.json').chmod(stat.S_IREAD)
    summary={'status':'CHECKPOINT_FROZEN_AWAITING_COMPARISON','experiments':experiments,
             'selection':lock,'e3_loss_choice':e3_loss,'top_two_macro_f1_gap':gap,
             'safety_macro_f1_negligible_definition':0.02,'stability_repeat_required':gap<.02,
             'truncated_rows':truncated,'smoke_test':smoke,'pipeline_checkpoint_replaced':False}
    write_json(run_dir/'comparison_summary.json',summary)
    print('OFFICIAL_CHECKPOINT',official,flush=True)
    print('COMPARISON_SUMMARY',run_dir/'comparison_summary.json',flush=True)

if __name__=='__main__': main()
