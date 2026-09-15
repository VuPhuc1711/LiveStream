"""PhoBERT: train/validation only; place this file beside main.py.

Run with .venv-nlp/Scripts/python.exe train_phobert.py
Requires torch (CUDA), transformers, numpy, scikit-learn, underthesea.
All model parameters are fine-tuned. No keyword rules or label-derived
metadata enter the text encoder. The test file is never opened here.

Underthesea is a prototype segmentation choice, different from the original
PhoBERT RDRSegmenter pipeline. Use identical preprocessing during inference.
This small synthetic dataset supports a workflow demo, not a production claim.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import random
import unicodedata
from collections import Counter
from datetime import datetime
from itertools import islice
from pathlib import Path

LABELS = ["KHONG_CANH_BAO", "CAN_XAC_MINH", "CANH_BAO"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
MODEL_NAME = "vinai/phobert-base"


def normalize_text(text):
    # Preserve case, accents, punctuation and negations.
    return " ".join(unicodedata.normalize("NFC", text).split())


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"id", "text", "label", "family_id", "review_status"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: thieu cot {required - set(reader.fieldnames or [])}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: khong co du lieu")
    seen_ids, seen_texts = set(), set()
    for row in rows:
        if None in row or any(row.get(k) is None for k in required):
            raise ValueError(f"{path}: dong CSV sai so cot")
        for key in ("id", "label", "family_id", "review_status"):
            row[key] = row[key].strip()
        text = normalize_text(row["text"])
        if not row["id"] or not row["family_id"] or not text:
            raise ValueError(f"{path}: id/family_id/text rong")
        if row["label"] not in LABEL2ID:
            raise ValueError(f"{row['id']}: nhan khong hop le: {row['label']}")
        if row["review_status"] not in {"DA_DUYET", "DA_DUYET_AI"}:
            raise ValueError(f"{row['id']}: chua duyet")
        key = text.casefold()
        if row["id"] in seen_ids or key in seen_texts:
            raise ValueError(f"{path}: trung id hoac noi dung: {row['id']}")
        seen_ids.add(row["id"])
        seen_texts.add(key)
    if set(row["label"] for row in rows) != set(LABELS):
        raise ValueError(f"{path}: can co du ca 3 nhan")
    return rows


def check_separation(train, validation):
    for key in ("id", "family_id"):
        overlap = {r[key] for r in train} & {r[key] for r in validation}
        if overlap:
            raise ValueError(f"Train/validation trung {key}: {sorted(overlap)}")
    texts = lambda rows: {normalize_text(r["text"]).casefold() for r in rows}
    if texts(train) & texts(validation):
        raise ValueError("Train/validation trung noi dung")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "splits")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.grad_accum) < 1 or not 8 <= args.max_length <= 256:
        parser.error("epochs/batch-size/grad-accum >= 1; max-length tu 8 den 256")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr phai la so duong huu han")

    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
    from underthesea import word_tokenize

    if not torch.cuda.is_available():
        raise RuntimeError("Chua co CUDA. Hay dung Python trong .venv-nlp.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    paths = {name: args.data_dir / f"{name}.csv" for name in ("train", "validation")}
    rows = {name: read_rows(path) for name, path in paths.items()}
    check_separation(rows["train"], rows["validation"])
    for name, items in rows.items():
        print(f"{name}: {len(items)} cau | {dict(Counter(r['label'] for r in items))}", flush=True)

    print("Dang tai tokenizer PhoBERT (lan dau can Internet)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    encoded, truncation = {}, {}
    for name, items in rows.items():
        samples, truncated = [], 0
        for row in items:
            segmented = word_tokenize(normalize_text(row["text"]), format="text")
            raw = tokenizer(segmented, truncation=False, return_token_type_ids=False)
            truncated += int(len(raw["input_ids"]) > args.max_length)
            feature = tokenizer(segmented, truncation=True, max_length=args.max_length,
                                return_token_type_ids=False)
            feature["labels"] = LABEL2ID[row["label"]]
            samples.append(feature)
        encoded[name], truncation[name] = samples, truncated
        print(f"{name}: {truncated} cau vuot {args.max_length} tokens va bi cat", flush=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    loaders = {
        name: DataLoader(samples, batch_size=args.batch_size, shuffle=(name == "train"),
                         collate_fn=collator, num_workers=0)
        for name, samples in encoded.items()
    }

    print("Dang tai PhoBERT va tao dau phan loai 3 nhan...", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, id2label=dict(enumerate(LABELS)), label2id=LABEL2ID,
        problem_type="single_label_classification", attn_implementation="eager",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, foreach=False)
    scaler = torch.amp.GradScaler("cuda")

    run_dir = root / "models" / ("phobert_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run_dir.mkdir(parents=True, exist_ok=False)
    best_dir = run_dir / "best"
    metadata = {
        "status": "running", "model": MODEL_NAME, "labels": LABELS,
        "hyperparameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "preprocessing": {"normalization": "NFC and collapse whitespace; preserve case and punctuation",
                          "segmenter": "underthesea.word_tokenize(format='text')",
                          "max_length": args.max_length,
                          "note": "Use the same preprocessing during inference; different from original RDRSegmenter."},
        "versions": {p: importlib.metadata.version(p) for p in
                     ("torch", "numpy", "transformers", "underthesea", "scikit-learn")},
        "data_sha256": {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in paths.items()},
        "data_ids": {n: [r["id"] for r in rr] for n, rr in rows.items()},
        "truncated_rows": truncation, "selection": "highest validation macro F1; tie -> lower validation loss",
        "test_used": False,
        "limitation": "Small synthetic development data, labels partly AI-reviewed; not independent real-world evaluation.",
    }
    write_json(run_dir / "run_info.json", metadata)
    print("Thu muc ket qua:", run_dir, flush=True)
    history, best_key, best_epoch = [], (-1.0, -float("inf")), None

    for epoch in range(1, args.epochs + 1):
        model.train()
        iterator = iter(loaders["train"])
        total_loss, seen, update = 0.0, 0, 0
        while True:
            group = list(islice(iterator, args.grad_accum))
            if not group:
                break
            group_size = sum(len(b["labels"]) for b in group)
            optimizer.zero_grad(set_to_none=True)
            for batch in group:
                batch = {k: v.to("cuda") for k, v in batch.items()}
                count = len(batch["labels"])
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = model(**batch)
                    loss = output.loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Train loss NaN/Inf; dung de kiem tra.")
                total_loss += loss.item() * count
                seen += count
                # Weight the final (possibly smaller) accumulation group correctly.
                scaler.scale(loss * (count / group_size)).backward()
                del output, loss, batch
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            update += 1
            print(f"Epoch {epoch}/{args.epochs} | update {update} | train loss {total_loss / seen:.4f}", flush=True)
        optimizer.zero_grad(set_to_none=True)

        model.eval()
        truth, predictions, val_loss = [], [], 0.0
        with torch.no_grad():
            for batch in loaders["validation"]:
                batch = {k: v.to("cuda") for k, v in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = model(**batch)
                if not torch.isfinite(output.loss) or not torch.isfinite(output.logits).all():
                    raise RuntimeError("Validation NaN/Inf; dung de kiem tra.")
                val_loss += output.loss.item() * len(batch["labels"])
                truth.extend(batch["labels"].cpu().tolist())
                predictions.extend(output.logits.argmax(dim=-1).cpu().tolist())
                del output, batch
        val_loss /= len(truth)
        accuracy = float(accuracy_score(truth, predictions))
        macro_f1 = float(f1_score(truth, predictions, labels=[0, 1, 2], average="macro", zero_division=0))
        record = {"epoch": epoch, "train_loss": total_loss / seen, "validation_loss": val_loss,
                  "validation_accuracy": accuracy, "validation_macro_f1": macro_f1}
        history.append(record)
        write_json(run_dir / "history.json", history)
        print(f"VALIDATION | loss={val_loss:.4f} | accuracy={accuracy:.4f} | macro F1={macro_f1:.4f}", flush=True)
        key = (macro_f1, -val_loss)
        if key > best_key:
            best_key, best_epoch = key, epoch
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            write_json(best_dir / "preprocessing.json", metadata["preprocessing"])
            report = classification_report(truth, predictions, labels=[0, 1, 2],
                                           target_names=LABELS, zero_division=0, output_dict=True)
            write_json(run_dir / "best_validation_report.json", {
                **record, "classification_report": report,
                "confusion_matrix": confusion_matrix(truth, predictions, labels=[0, 1, 2]).tolist(),
                "matrix_label_order": LABELS, "matrix_rows": "true", "matrix_columns": "predicted",
            })
            with (run_dir / "validation_predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["id", "text", "label", "prediction", "correct"])
                writer.writeheader()
                for row, pred in zip(rows["validation"], predictions):
                    writer.writerow({"id": row["id"], "text": row["text"], "label": row["label"],
                                     "prediction": LABELS[pred], "correct": row["label"] == LABELS[pred]})
            print("Da luu mo hinh tot nhat hien tai.", flush=True)

    metadata.update(status="completed", best_epoch=best_epoch, best_validation_macro_f1=best_key[0])
    write_json(run_dir / "run_info.json", metadata)
    print(f"\nHOAN TAT | Best epoch: {best_epoch} | Validation macro F1: {best_key[0]:.4f}")
    print("Mo hinh:", best_dir)
    print("Bang du doan:", run_dir / "validation_predictions.csv")
    print("Chua danh gia test. Day la ket qua noi bo tren du lieu mo phong.")


if __name__ == "__main__":
    main()
