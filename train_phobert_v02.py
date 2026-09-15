"""PhoBERT v02: train/validation only, initialized from vinai/phobert-base.

Run with .venv-nlp/Scripts/python.exe -B -X utf8 train_phobert_v02.py
Requires torch (CUDA), transformers, numpy, scikit-learn, underthesea.
All model parameters are fine-tuned. No keyword rules or label-derived
metadata enter the text encoder. Final test is read only as bytes to verify
its frozen hash; no final-test CSV rows are parsed, tokenized or predicted.

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
import os
import random
import stat
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

LABELS = ["KHONG_CANH_BAO", "CAN_XAC_MINH", "CANH_BAO"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
MODEL_NAME = "vinai/phobert-base"
BASE_REVISION = "01daacda68afe13d83023d16ec647239e344a1e6"


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


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_final_test_lock(root):
    lock_path = root / "reports" / "final_test_v03.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8-sig"))
    final_path = root / "data" / "final_test_v03_chua_duyet.csv"
    if Path(lock["path"]).resolve() != final_path.resolve():
        raise ValueError("Final-test lock points to a different file.")
    if lock.get("status") != "FROZEN_USER_REVIEWED" or not lock.get("user_confirmed_labels_final"):
        raise ValueError("Final test has not been frozen with user approval.")
    if sha256(final_path) != lock["sha256"]:
        raise ValueError("Final-test bytes differ from the frozen hash.")
    if not final_path.stat().st_file_attributes & stat.FILE_ATTRIBUTE_READONLY:
        raise ValueError("Final test must remain read-only.")
    return lock


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "splits_v02")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.grad_accum) < 1 or not 8 <= args.max_length <= 256:
        parser.error("epochs/batch-size/grad-accum >= 1; max-length tu 8 den 256")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr phai la so duong huu han")
    if args.data_dir.resolve() != (root / "data" / "splits_v02").resolve() or args.seed != 42:
        parser.error("Trainer v02 nay chi dung data/splits_v02 va seed 42.")

    final_lock = verify_final_test_lock(root)
    protected_paths = (
        [root / "data" / name for name in ("dataset_livestream_v01.csv", "dataset_livestream_v02.csv",
                                           "final_test_v03_chua_duyet.csv")]
        + list((root / "data" / "splits").glob("*.csv"))
        + list((root / "data" / "splits_v02").glob("*.csv"))
        + list((root / "nlp").glob("*.py"))
        + [root / name for name in ("main.py", "train_phobert.py", "reports/final_test_v03.lock.json")]
        + [path for path in (root / "models" / "phobert_20260909_182641_815167").rglob("*") if path.is_file()]
    )
    protected_before = {str(path.relative_to(root)): sha256(path) for path in protected_paths}
    print("Final test locked: hash-only integrity checks; no test examples used.", flush=True)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
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
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print(f"GPU memory: {free_bytes / 1024**3:.2f} / {total_bytes / 1024**3:.2f} GiB free", flush=True)

    paths = {name: args.data_dir / f"{name}.csv" for name in ("train", "validation")}
    rows = {name: read_rows(path) for name, path in paths.items()}
    check_separation(rows["train"], rows["validation"])
    preparation = json.loads((root / "reports" / "data_preparation_v02_v03" / "report.json").read_text(encoding="utf-8"))
    for name, expected_count in (("train", 120), ("validation", 30)):
        if len(rows[name]) != expected_count or any(row.get("split") != name for row in rows[name]):
            raise ValueError(f"Expected {expected_count} rows in split {name}.")
        expected_hash = preparation["output_sha256"][str(paths[name].relative_to(root))]
        if sha256(paths[name]) != expected_hash:
            raise ValueError(f"Split {name} differs from the prepared v02 data.")
        if any(row["id"].startswith("FTV03_") or row["family_id"].startswith("FTV03_") for row in rows[name]):
            raise ValueError("Final-test identifier found in training/validation input.")
    for name, items in rows.items():
        print(f"{name}: {len(items)} cau | {dict(Counter(r['label'] for r in items))}", flush=True)

    print("Loading tokenizer from the cached, pinned vinai/phobert-base revision...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, revision=BASE_REVISION, use_fast=False, local_files_only=True
    )
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
    model, loading_info = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, id2label=dict(enumerate(LABELS)), label2id=LABEL2ID,
        problem_type="single_label_classification", attn_implementation="eager",
        revision=BASE_REVISION, local_files_only=True, use_safetensors=False,
        output_loading_info=True,
    )
    if loading_info.get("error_msgs") or loading_info.get("mismatched_keys"):
        raise RuntimeError(f"Unexpected base-model loading issue: {loading_info}")
    missing = loading_info.get("missing_keys", [])
    if any(not key.startswith("classifier.") for key in missing):
        raise RuntimeError(f"Pretrained encoder weights are missing: {missing}")
    from huggingface_hub import hf_hub_download
    base_weights_path = Path(hf_hub_download(
        MODEL_NAME, filename="pytorch_model.bin", revision=BASE_REVISION, local_files_only=True
    ))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, foreach=False)
    scaler = torch.amp.GradScaler("cuda")

    run_dir = root / "models" / ("phobert_v02_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run_dir.mkdir(parents=True, exist_ok=False)
    best_dir = run_dir / "best"
    metadata = {
        "status": "running", "model": MODEL_NAME, "labels": LABELS,
        "base_revision": BASE_REVISION, "base_weights_sha256": sha256(base_weights_path),
        "initialization": "Fresh pretrained vinai/phobert-base encoder and new 3-label classification head; no v01 checkpoint",
        "base_loading_info": {"missing_keys": sorted(missing), "unexpected_keys": sorted(loading_info.get("unexpected_keys", []))},
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
        "final_test": {"sha256": final_lock["sha256"], "user_confirmed_labels_final": True,
                       "access": "hash-only integrity verification; no CSV parsing or model inputs",
                       "used_for_training": False, "used_for_selection": False, "evaluated": False},
        "training_configuration": {"optimizer": "AdamW", "weight_decay": 0.01, "gradient_clip_norm": 1.0,
                                   "mixed_precision": "float16 with GradScaler", "gradient_checkpointing": True,
                                   "effective_batch_size": args.batch_size * args.grad_accum, "scheduler": None},
        "protected_file_sha256": protected_before, "trainer_sha256": sha256(Path(__file__).resolve()),
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
        truth, predictions, probabilities, val_loss = [], [], [], 0.0
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
                probabilities.extend(output.logits.float().softmax(dim=-1).cpu().tolist())
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
                writer = csv.DictWriter(stream, fieldnames=["id", "text", "label", "prediction", "correct", "confidence",
                                                           *[f"p_{label}" for label in LABELS]])
                writer.writeheader()
                for row, pred, scores in zip(rows["validation"], predictions, probabilities):
                    writer.writerow({"id": row["id"], "text": row["text"], "label": row["label"],
                                     "prediction": LABELS[pred], "correct": row["label"] == LABELS[pred],
                                     "confidence": scores[pred], **{f"p_{label}": scores[i] for i, label in enumerate(LABELS)}})
            print("Da luu mo hinh tot nhat hien tai.", flush=True)

    verify_final_test_lock(root)
    protected_after = {str(path.relative_to(root)): sha256(path) for path in protected_paths}
    if protected_after != protected_before:
        raise RuntimeError("A protected file changed during training; checkpoint is not marked final.")
    best_record = next(record for record in history if record["epoch"] == best_epoch)
    frozen_at = datetime.now(timezone.utc).isoformat()
    for path in best_dir.iterdir():
        if path.is_file():
            path.chmod(path.stat().st_mode & ~stat.S_IWRITE)
    metadata.update(status="completed", best_epoch=best_epoch, best_validation_macro_f1=best_key[0],
                    best_validation_loss=best_record["validation_loss"],
                    best_validation_accuracy=best_record["validation_accuracy"],
                    protected_files_unchanged=True, best_checkpoint_frozen_at_utc=frozen_at)
    write_json(run_dir / "run_info.json", metadata)
    locked_artifacts = sorted(path for path in best_dir.iterdir() if path.is_file()) + [
        run_dir / name for name in ("run_info.json", "history.json", "best_validation_report.json", "validation_predictions.csv")
    ]
    lock_path = run_dir / "checkpoint_lock.json"
    write_json(lock_path, {
        "status": "FROZEN", "frozen_at_utc": frozen_at, "checkpoint": str(best_dir),
        "base_model": MODEL_NAME, "base_revision": BASE_REVISION,
        "selection": metadata["selection"], "selected_validation_result": best_record,
        "artifact_sha256": {str(path.relative_to(run_dir)): sha256(path) for path in locked_artifacts},
        "final_test_sha256": final_lock["sha256"], "final_test_evaluated": False,
    })
    lock_path.chmod(lock_path.stat().st_mode & ~stat.S_IWRITE)
    print(f"\nHOAN TAT | Best epoch: {best_epoch} | Validation macro F1: {best_key[0]:.4f}")
    print("Mo hinh:", best_dir)
    print("Bang du doan:", run_dir / "validation_predictions.csv")
    print("Best checkpoint frozen:", lock_path)
    print("Final test v03 remains locked and has not been evaluated.")


if __name__ == "__main__":
    main()
