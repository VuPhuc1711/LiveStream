"""Đánh giá cuối v03 đúng một lần bằng checkpoint PhoBERT v02 đã chốt.

Chạy: .venv-nlp/Scripts/python.exe -B -X utf8 evaluate_final_v03.py
Kiểm tra bằng dữ liệu giả, không mở final test: thêm --self-test.
Không có tùy chọn đổi checkpoint, dữ liệu, ngưỡng hoặc chạy lại.
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
import sys
import tempfile
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "models/phobert_v02_20260914_162150_178711/best"
TEST_PATH = ROOT / "data/final_test_v03_chua_duyet.csv"
TEST_LOCK = ROOT / "reports/final_test_v03.lock.json"
CHECKPOINT_LOCK = CHECKPOINT.parent / "checkpoint_lock.json"
RUN_CLAIM = ROOT / "reports/.final_evaluation_v03.started.json"
EVALUATION_LOCK = ROOT / "final_evaluation_lock.json"
LABELS = ("KHONG_CANH_BAO", "CAN_XAC_MINH", "CANH_BAO")
METHODS = {"rules": "Rules", "phobert_v02": "PhoBERT v02", "hybrid": "HybridDetector"}
COLUMNS = ("id", "text", "label", "category", "reason", "family_id", "review_status", "needs_context")
EXPECTED_TEST_HASH = "7b6114a9e7d6e569fd9e28d09e3f655becd8297fed7771e5704741766523b253"
EXPECTED_MODEL_HASH = "fef1b0c901e60874d8d802afad1454878110dc17c587f01936375a6ca8663268"
ERROR_TYPES = {
    "false_review": "Đưa đi kiểm tra nhầm: KHONG_CANH_BAO → nhãn khác",
    "missed_review": "Bỏ sót sàng lọc: CAN_XAC_MINH/CANH_BAO → KHONG_CANH_BAO",
    "false_warning": "Cảnh báo mức cao nhầm: nhãn khác → CANH_BAO",
    "missed_warning": "Không giữ mức cảnh báo: CANH_BAO → nhãn khác",
    "warning_to_review": "Hạ xuống xác minh: CANH_BAO → CAN_XAC_MINH; vẫn được xem lại",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json_new(path, value):
    # Exclusive creation prevents overwriting reports or a concurrent run's claim.
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def assert_unused(claim=RUN_CLAIM, lock=EVALUATION_LOCK):
    if claim.exists() or lock.exists():
        raise RuntimeError(
            "Đánh giá cuối đã được bắt đầu hoặc hoàn tất. Chặn chạy lại; "
            "chỉ được đọc kết quả đã lưu, không xóa khóa để dự đoán lại."
        )


def claim_run(metadata, claim=RUN_CLAIM, lock=EVALUATION_LOCK):
    assert_unused(claim, lock)
    write_json_new(claim, metadata)


def normalized_key(text):
    return " ".join(unicodedata.normalize("NFC", text).split()).casefold()


def verify_frozen_inputs():
    frozen = read_json(CHECKPOINT_LOCK)
    reviewed = read_json(TEST_LOCK)
    require(frozen["status"] == "FROZEN", "Checkpoint chưa được chốt.")
    require(Path(frozen["checkpoint"]).resolve() == CHECKPOINT.resolve(), "Sai checkpoint trong bản khóa.")
    require(Path(reviewed["path"]).resolve() == TEST_PATH.resolve(), "Sai final test trong bản khóa.")
    require(reviewed["status"] == "FROZEN_USER_REVIEWED" and reviewed["user_confirmed_labels_final"],
            "Chưa có xác nhận chốt nhãn của người dùng.")
    require(reviewed["sha256"] == frozen["final_test_sha256"] == EXPECTED_TEST_HASH,
            "Hash final test trong bản khóa không khớp bản đã chốt.")
    protected = {}
    for relative, expected in frozen["artifact_sha256"].items():
        path = CHECKPOINT.parent / relative
        require(sha256(path) == expected, f"Artifact checkpoint đã thay đổi: {path}")
        protected[path.relative_to(ROOT).as_posix()] = expected
    checkpoint_files = {
        path.relative_to(CHECKPOINT).as_posix(): sha256(path)
        for path in sorted(CHECKPOINT.rglob("*")) if path.is_file()
    }
    expected_files = {
        Path(relative).relative_to("best").as_posix()
        for relative in frozen["artifact_sha256"] if Path(relative).parts[0] == "best"
    }
    require(set(checkpoint_files) == expected_files, "Danh sách file checkpoint đã thay đổi.")
    require(checkpoint_files["model.safetensors"] == EXPECTED_MODEL_HASH, "Không đúng trọng số v02 bắt buộc.")
    if os.name == "nt":
        require(TEST_PATH.stat().st_file_attributes & stat.FILE_ATTRIBUTE_READONLY,
                "Final test không còn thuộc tính ReadOnly.")
        require(all(path.stat().st_file_attributes & stat.FILE_ATTRIBUTE_READONLY
                    for path in CHECKPOINT.iterdir() if path.is_file()), "Checkpoint không còn ReadOnly.")
    training = read_json(CHECKPOINT.parent / "run_info.json")
    require(training["status"] == "completed" and training["model"] == "vinai/phobert-base",
            "Lần train v02 chưa hoàn tất hoặc sai model gốc.")
    require(not training["test_used"] and not training["final_test"]["evaluated"],
            "Thông tin train ghi nhận đã sử dụng final test.")
    for relative, expected in training["protected_file_sha256"].items():
        path = ROOT / relative
        require(sha256(path) == expected, f"File được bảo vệ đã thay đổi: {path}")
        protected[path.relative_to(ROOT).as_posix()] = expected
    require(sha256(TEST_PATH) == EXPECTED_TEST_HASH, "Nội dung final test không khớp bản khóa.")
    for path in (CHECKPOINT_LOCK, TEST_LOCK, Path(__file__).resolve()):
        protected[path.relative_to(ROOT).as_posix()] = sha256(path)
    canonical = json.dumps(checkpoint_files, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {
        "checkpoint": str(CHECKPOINT.resolve()),
        "checkpoint_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "checkpoint_hash_definition": "SHA256 of UTF-8 JSON {relative POSIX filename: file SHA256}, sorted keys, separators=(',', ':')",
        "checkpoint_file_sha256": checkpoint_files,
        "model_weights_sha256": EXPECTED_MODEL_HASH,
        "final_test": str(TEST_PATH.resolve()),
        "final_test_sha256": EXPECTED_TEST_HASH,
        "checkpoint_frozen_at_utc": frozen["frozen_at_utc"],
        "selected_epoch": frozen["selected_validation_result"]["epoch"],
        "labels_approved_by_user": True,
        "approval_manifest": str(TEST_LOCK),
        "protected_file_sha256_before": protected,
    }, training


def load_final_rows(training):
    with TEST_PATH.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(tuple(reader.fieldnames or ()) == COLUMNS, "Sai các cột final test.")
        rows = list(reader)
    require(len(rows) == 30, "Final test phải có đúng 30 câu.")
    require(all(None not in row and all(isinstance(row[k], str) and row[k].strip() for k in COLUMNS)
                for row in rows), "CSV thiếu giá trị hoặc sai số cột.")
    require({r["id"] for r in rows} == {f"FTV03_{i:03d}" for i in range(1, 31)}, "Sai ID final test.")
    require(len({r["family_id"] for r in rows}) == 30, "Trùng family_id final test.")
    require(len({normalized_key(r["text"]) for r in rows}) == 30, "Trùng text chuẩn hóa.")
    require(Counter(r["label"] for r in rows) == Counter({label: 10 for label in LABELS}), "Sai phân bố nhãn.")
    for split in ("train", "validation"):
        require(not {r["id"] for r in rows}.intersection(training["data_ids"][split]),
                f"ID final test có trong {split} đã train.")
    # CHUA_DUYET is preserved in the frozen CSV; TEST_LOCK records the subsequent user approval.
    return rows


def make_record(row, result):
    pho = result["phobert_result"]
    rules = result["rule_result"]
    require(set(pho["probabilities"]) == set(LABELS), "Sai tập nhãn dự đoán.")
    require(all(math.isfinite(v) and 0 <= v <= 1 for v in pho["probabilities"].values()), "Xác suất không hợp lệ.")
    require(math.isclose(sum(pho["probabilities"].values()), 1, abs_tol=1e-5), "Tổng xác suất khác 1.")
    require(pho["label"] == max(pho["probabilities"], key=pho["probabilities"].get), "Nhãn PhoBERT sai argmax.")
    require(pho["confidence"] == pho["probabilities"][pho["label"]], "Sai confidence PhoBERT.")
    require(result["final_label"] in LABELS and
            result["final_confidence"] == pho["probabilities"][result["final_label"]], "Sai confidence Hybrid.")
    keywords = list(dict.fromkeys(item["keyword"] for item in rules["findings"]))
    rule_explanation = (
        "Rules khớp các cụm từ: " + ", ".join(keywords) + "; chuyển CAN_XAC_MINH."
        if rules["needs_review"] else "Rules không khớp cụm từ nào; chọn KHONG_CANH_BAO."
    ) + " Rules không cung cấp confidence và không tự kết luận CANH_BAO."
    predictions = {
        "rules": {"label": "CAN_XAC_MINH" if rules["needs_review"] else "KHONG_CANH_BAO",
                  "confidence": None, "probabilities": None, "explanation": rule_explanation},
        "phobert_v02": {**pho, "explanation":
            f"PhoBERT v02 chọn {pho['label']} vì có xác suất softmax lớn nhất ({pho['confidence']:.2%}). "
            "Đây là mô tả cách chọn nhãn, không phải giải thích nguyên nhân ngôn ngữ bên trong model."},
        "hybrid": {"label": result["final_label"], "confidence": result["final_confidence"],
                   "probabilities": pho["probabilities"], "explanation": result["explanation"]},
    }
    return {"id": row["id"], "text": row["text"], "true_label": row["label"],
            "family_id": row["family_id"], "predictions": predictions, "hybrid_result": result}


def collect_predictions(rows, detector, journal_path):
    records = []
    with journal_path.open("x", encoding="utf-8", newline="\n") as journal:
        for row in rows:
            # The single call evaluates both components once; reuse them for all three baselines.
            result = detector.predict(row["text"])
            # Persist raw output before processing, so reporting errors never require inference again.
            journal.write(json.dumps({"row": row, "hybrid_result": result}, ensure_ascii=False, allow_nan=False) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
            records.append(make_record(row, result))
            print(f"Đã lưu dự đoán {len(records):02d}/{len(rows)}: {row['id']}", flush=True)
    return records


def error_types(truth, predicted):
    checks = {
        "false_review": truth == LABELS[0] and predicted != LABELS[0],
        "missed_review": truth != LABELS[0] and predicted == LABELS[0],
        "false_warning": truth != LABELS[2] and predicted == LABELS[2],
        "missed_warning": truth == LABELS[2] and predicted != LABELS[2],
        "warning_to_review": truth == LABELS[2] and predicted == LABELS[1],
    }
    return [name for name, matches in checks.items() if matches]


def evaluate_method(records, method):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    truth = [r["true_label"] for r in records]
    predicted = [r["predictions"][method]["label"] for r in records]
    scores = classification_report(truth, predicted, labels=list(LABELS), output_dict=True, zero_division=0)
    errors = []
    for record in records:
        prediction = record["predictions"][method]
        if record["true_label"] != prediction["label"]:
            errors.append({"id": record["id"], "text": record["text"], "true_label": record["true_label"],
                           "predicted_label": prediction["label"], "confidence": prediction["confidence"],
                           "explanation": prediction["explanation"],
                           "error_types": error_types(record["true_label"], prediction["label"])})
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_precision": scores["macro avg"]["precision"],
        "macro_recall": scores["macro avg"]["recall"],
        "macro_f1": scores["macro avg"]["f1-score"],
        "correct": len(records) - len(errors), "total": len(records),
        "per_label": {label: {"precision": scores[label]["precision"], "recall": scores[label]["recall"],
                              "f1": scores[label]["f1-score"], "support": int(scores[label]["support"])}
                      for label in LABELS},
        "confusion_matrix": confusion_matrix(truth, predicted, labels=list(LABELS)).tolist(),
        "matrix_label_order": list(LABELS), "matrix_rows": "true", "matrix_columns": "predicted",
        "errors": errors,
        "error_groups": {name: [r["id"] for r in errors if name in r["error_types"]] for name in ERROR_TYPES},
    }


def markdown_cell(value):
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def render_report(report):
    provenance = report["provenance"]
    lines = [
        "# Đánh giá cuối final test v03", "",
        f"Checkpoint: `{provenance['checkpoint']}` (best epoch {provenance['selected_epoch']}).",
        f"Tập test: `{provenance['final_test']}`; 30 câu, 10 câu mỗi nhãn.",
        f"Thời gian UTC: {report['started_at_utc']} → {report['finished_at_utc']}.",
        "Đã chạy đúng một lượt: 30 lần Hybrid.predict, gồm 30 lần PhoBERT và 30 lần Rules; tái sử dụng kết quả cho cả ba phương pháp.",
        "Không train, chọn epoch, thay nhãn, sửa Rules, checkpoint hoặc ngưỡng sau khi xem kết quả.", "",
        "Nhãn đã được người dùng duyệt và chốt trong reports/final_test_v03.lock.json. "
        "Tên file và giá trị CHUA_DUYET trong CSV được giữ nguyên theo bản khóa.", "",
        "## Quy ước đã cố định trước khi chạy", "",
        "Rules: khớp → CAN_XAC_MINH; không khớp → KHONG_CANH_BAO. Rules không dự đoán CANH_BAO; "
        "confidence không có (JSON null, CSV để trống), không gán điểm giả.",
        "PhoBERT: lấy nhãn có xác suất softmax cao nhất. Giữ preprocessing và tokenizer trong checkpoint.",
        "Hybrid: CANH_BAO nếu PhoBERT chọn nhãn đó với confidence ≥ 0.55; KHONG_CANH_BAO nếu "
        "PhoBERT chọn nhãn đó với confidence ≥ 0.65, kể cả Rules khớp; còn lại CAN_XAC_MINH.",
        "Confidence PhoBERT chưa được hiệu chỉnh. Confidence Hybrid là xác suất PhoBERT của nhãn cuối; "
        "ba xác suất ghi cho Hybrid cũng là của PhoBERT, không phải phân phối xác suất mới của Hybrid.",
        "Macro-F1 là trung bình F1 của đủ ba nhãn. Các phép chia không xác định được tính bằng 0 (zero_division=0).", "",
        "## Chỉ số tổng hợp", "",
        "| Phương pháp | Đúng/Tổng | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, name in METHODS.items():
        result = report["methods"][method]
        lines.append(f"| {name} | {result['correct']}/{result['total']} | {result['accuracy']:.2%} | "
                     f"{result['macro_precision']:.2%} | {result['macro_recall']:.2%} | {result['macro_f1']:.2%} |")
    for method, name in METHODS.items():
        result = report["methods"][method]
        lines += ["", f"## {name}", "", "| Nhãn | Precision | Recall | F1 | Số câu thật |",
                  "|---|---:|---:|---:|---:|"]
        for label, scores in result["per_label"].items():
            lines.append(f"| {label} | {scores['precision']:.2%} | {scores['recall']:.2%} | {scores['f1']:.2%} | {scores['support']} |")
        lines += ["", "### Confusion matrix", "", "Hàng là nhãn thật; cột là nhãn dự đoán.", "",
                  "| Thật \\ Dự đoán | " + " | ".join(LABELS) + " |", "|---|---:|---:|---:|"]
        for label, values in zip(LABELS, result["confusion_matrix"]):
            lines.append("| " + label + " | " + " | ".join(map(str, values)) + " |")
        lines += ["", "### Dự đoán nhầm và bỏ sót", "", "Các nhóm có thể giao nhau; không cộng số nhóm thành tổng lỗi.", ""]
        for tag, description in ERROR_TYPES.items():
            ids = result["error_groups"][tag]
            lines.append(f"- {description}: {len(ids)} câu" + (" — " + ", ".join(ids) if ids else "") + ".")
        lines += ["", f"### Toàn bộ câu sai ({len(result['errors'])})", ""]
        if not result["errors"]:
            lines.append("Không có câu dự đoán sai.")
        for error in result["errors"]:
            confidence = "không có (Rules)" if error["confidence"] is None else f"{error['confidence']:.6f} ({error['confidence']:.2%})"
            lines += [f"**{error['id']}** — thật: {error['true_label']}; dự đoán: {error['predicted_label']}; confidence: {confidence}.",
                      "", markdown_cell(error["text"]), "", "Giải thích: " + markdown_cell(error["explanation"]), ""]
    lines += ["## Tính toàn vẹn và giới hạn diễn giải", "",
              f"SHA-256 trọng số: `{provenance['model_weights_sha256']}`.",
              f"SHA-256 toàn bộ checkpoint: `{provenance['checkpoint_sha256']}`; cách tính được ghi rõ trong report.json.",
              f"SHA-256 final test: `{provenance['final_test_sha256']}`.",
              f"Đối chiếu trước/sau: {len(provenance['protected_file_sha256_before'])} file được bảo vệ không đổi.",
              "Đây là 30 câu mô phỏng đã chốt nhãn, chưa đại diện cho toàn bộ lời nói livestream thực tế. "
              "Mỗi câu làm Accuracy thay đổi khoảng 3.33 điểm phần trăm; mỗi nhãn chỉ có 10 câu.",
              "Rà soát dữ liệu trước khi train đã ghi nhận 13 câu v03 gần nghĩa cao/rất cao với v02 "
              "(reports/data_preparation_v02_v03/README.md). Đây là nhận xét định tính từ trước, "
              "không phải kiểm tra mới bằng model. Không trùng ID/text không chứng minh độc lập hoàn toàn về nội dung.",
              "Điểm Rules ở bài toán ba nhãn bị giới hạn bởi chức năng sàng lọc hai đầu ra sẵn có. "
              "Hạ CANH_BAO xuống CAN_XAC_MINH vẫn chuyển người kiểm duyệt, khác với bỏ sót hoàn toàn về KHONG_CANH_BAO.",
              "Kết quả này chỉ để báo cáo đánh giá cuối; không dùng để chỉnh model, Rules hoặc ngưỡng.", ""]
    return "\n".join(lines)


def write_predictions(path, records):
    fields = ["id", "text", "true_label", "family_id", "method", "predicted_label", "correct", "confidence",
              *[f"probability_{label}" for label in LABELS], "probability_source", "explanation", "error_types", "rule_keywords"]
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for method, prediction in record["predictions"].items():
                probs = prediction["probabilities"] or {}
                writer.writerow({
                    "id": record["id"], "text": record["text"], "true_label": record["true_label"],
                    "family_id": record["family_id"], "method": method, "predicted_label": prediction["label"],
                    "correct": record["true_label"] == prediction["label"], "confidence": prediction["confidence"],
                    **{f"probability_{label}": probs.get(label) for label in LABELS},
                    "probability_source": "" if method == "rules" else "PhoBERT v02 softmax",
                    "explanation": prediction["explanation"],
                    "error_types": ";".join(error_types(record["true_label"], prediction["label"])),
                    "rule_keywords": ";".join(dict.fromkeys(x["keyword"] for x in record["hybrid_result"]["rule_result"]["findings"])),
                })


def self_test():
    """No final-test access or model construction; test metrics, export and the one-shot guard."""
    rows = [{"id": f"FAKE_{i}", "text": f'Câu giả {i}, "nháy" | xuống\ndòng.', "label": label, "family_id": str(i)}
            for i, label in enumerate((LABELS[0], LABELS[0], LABELS[1], LABELS[2]))]

    class FakeDetector:
        calls = 0

        def predict(self, text):
            label = (LABELS[0], LABELS[1], LABELS[2], LABELS[2])[self.calls]
            self.calls += 1
            probabilities = {name: 0.8 if name == label else 0.1 for name in LABELS}
            return {"final_label": label, "final_confidence": 0.8,
                    "phobert_result": {"label": label, "confidence": 0.8, "probabilities": probabilities},
                    "rule_result": {"needs_review": False, "findings": []}, "explanation": "Giải thích giả lập."}

    with tempfile.TemporaryDirectory(prefix="final_v03_synthetic_") as temp:
        base = Path(temp)
        claim, lock = base / "claim.json", base / "lock.json"
        claim_run({"synthetic": True}, claim, lock)
        detector = FakeDetector()
        records = collect_predictions(rows, detector, base / "journal.jsonl")
        require(detector.calls == len(rows), "Mỗi câu phải gọi detector đúng một lần.")
        for blocked_claim, blocked_lock in ((claim, lock), (base / "unused.json", claim)):
            try:
                claim_run({}, blocked_claim, blocked_lock)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Khóa không chặn chạy lại.")
        metrics = {method: evaluate_method(records, method) for method in METHODS}
        require(metrics["hybrid"]["confusion_matrix"] == [[1, 1, 0], [0, 0, 1], [0, 0, 1]], "Đảo hàng/cột ma trận.")
        require(math.isclose(metrics["hybrid"]["accuracy"], 0.5), "Sai Accuracy.")
        require(math.isclose(metrics["hybrid"]["macro_f1"], 4 / 9), "Sai Macro-F1.")
        require(metrics["rules"]["per_label"][LABELS[2]]["f1"] == 0, "Sai zero_division.")
        require(all(e["confidence"] is None for e in metrics["rules"]["errors"]), "Rules không có confidence.")
        require(metrics["hybrid"]["error_groups"]["false_warning"] == ["FAKE_2"], "Sai nhóm lỗi.")
        write_predictions(base / "predictions.csv", records)
        with (base / "predictions.csv").open(encoding="utf-8", newline="") as stream:
            exported = list(csv.DictReader(stream))
        require(len(exported) == 12 and exported[0]["text"] == rows[0]["text"], "Sai CSV roundtrip.")
        require(exported[0]["confidence"] == "", "CSV Rules phải để trống confidence.")
        fake_report = {"provenance": {"checkpoint": "FAKE", "selected_epoch": 1, "final_test": "FAKE",
                       "model_weights_sha256": "FAKE", "checkpoint_sha256": "FAKE", "final_test_sha256": "FAKE",
                       "protected_file_sha256_before": {}}, "started_at_utc": "FAKE", "finished_at_utc": "FAKE", "methods": metrics}
        rendered = render_report(fake_report)
        require("FAKE_2" in rendered and "Giải thích giả lập." in rendered, "Thiếu chi tiết lỗi Markdown.")
        write_json_new(base / "report.json", fake_report)
        require(read_json(base / "report.json")["methods"] == metrics, "Sai JSON roundtrip.")
    print("SELF-TEST PASS: metrics, matrix orientation, errors, CSV/JSON/Markdown, single call per row, one-shot guard. No final test/model accessed.")


def main():
    assert_unused()
    started_at, started_clock = now(), time.perf_counter()
    print("Kiểm tra hash và trạng thái khóa trước khi nạp model...", flush=True)
    provenance, training = verify_frozen_inputs()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import numpy as np
    import torch
    from nlp.hybrid_detector import HybridDetector, SAFE_THRESHOLD, WARNING_THRESHOLD
    from nlp.keyword_detector import KeywordDetector
    from nlp.phobert_detector import PhoBertDetector

    require((WARNING_THRESHOLD, SAFE_THRESHOLD) == (0.55, 0.65), "Ngưỡng Hybrid đã thay đổi.")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    phobert = PhoBertDetector(checkpoint_path=CHECKPOINT)
    require(phobert.checkpoint_path == CHECKPOINT.resolve() and not phobert.model.training,
            "Detector nạp sai checkpoint hoặc chưa ở eval mode.")
    detector = HybridDetector(phobert_detector=phobert, rule_detector=KeywordDetector())
    configuration = {
        "labels": list(LABELS), "seed": 42, "warning_threshold": WARNING_THRESHOLD, "safe_threshold": SAFE_THRESHOLD,
        "rules_mapping": {"match": "CAN_XAC_MINH", "no_match": "KHONG_CANH_BAO", "confidence": None},
        "preprocessing": read_json(CHECKPOINT / "preprocessing.json"), "local_files_only": True,
        "device": str(phobert.device), "dtype": str(next(phobert.model.parameters()).dtype),
        "device_name": torch.cuda.get_device_name(phobert.device) if phobert.device.type == "cuda" else "CPU",
        "attention_implementation": phobert.model.config._attn_implementation,
        "model_eval_mode": True, "inference_mode": True, "batch_size": 1,
        "python": sys.version, "python_executable": sys.executable,
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "underthesea", "scikit-learn", "numpy")},
        "zero_division": 0,
    }
    output = ROOT / "reports" / ("final_test_v03_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    claim_run({"status": "STARTED_DO_NOT_RERUN", "claimed_at_utc": now(), "started_at_utc": started_at,
               "report_directory": str(output), "provenance": provenance, "configuration": configuration})
    # From this point even an interrupted run stays consumed. Reporting may use the saved journal, never rerun inference.
    output.mkdir(parents=True, exist_ok=False)
    try:
        rows = load_final_rows(training)
        records = collect_predictions(rows, detector, output / "inference_journal.jsonl")
        after = {relative: sha256(ROOT / relative) for relative in provenance["protected_file_sha256_before"]}
        require(after == provenance["protected_file_sha256_before"], "File bảo vệ đã thay đổi trong khi đánh giá.")
        provenance["protected_file_sha256_after"] = after
        provenance["protected_files_unchanged"] = True
        report = {
            "schema_version": 1, "status": "COMPLETED", "started_at_utc": started_at, "finished_at_utc": now(),
            "duration_seconds": time.perf_counter() - started_clock, "provenance": provenance,
            "configuration": configuration, "report_directory": str(output),
            "dataset": {"rows": len(rows), "label_counts": dict(Counter(r["label"] for r in rows)),
                        "review_status_values_preserved": dict(Counter(r["review_status"] for r in rows)),
                        "labels_approved_by_user": True, "ids_absent_from_recorded_train_and_validation": True},
            "inference_counts": {"hybrid_predict": len(records), "phobert_predict": len(records), "rules_detect": len(records)},
            "error_type_definitions": ERROR_TYPES,
            "methods": {method: evaluate_method(records, method) for method in METHODS}, "predictions": records,
            "used_for_training_or_tuning": False,
            "limitations": ["30 synthetic, user-reviewed examples; 10 per class; not a real-world representative benchmark.",
                            "Earlier qualitative audit flagged 13 examples as highly semantically similar to v02; no new analysis or tuning performed.",
                            "Rules has no CANH_BAO output or calibrated confidence; Hybrid confidence is the PhoBERT probability of its final label."],
        }
        write_predictions(output / "predictions.csv", records)
        write_json_new(output / "report.json", report)
        markdown = render_report(report)
        with (output / "report.md").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(markdown)
        lock = {
            "schema_version": 1, "status": "COMPLETED_LOCKED", "checkpoint": provenance["checkpoint"],
            "checkpoint_sha256": provenance["checkpoint_sha256"],
            "checkpoint_hash_definition": provenance["checkpoint_hash_definition"],
            "checkpoint_file_sha256": provenance["checkpoint_file_sha256"],
            "model_weights_sha256": provenance["model_weights_sha256"], "final_test": provenance["final_test"],
            "final_test_sha256": provenance["final_test_sha256"], "started_at_utc": started_at,
            "finished_at_utc": report["finished_at_utc"], "locked_at_utc": now(),
            "duration_seconds": report["duration_seconds"], "report_directory": str(output),
            "report_path": str(output / "report.md"), "claim_path": str(RUN_CLAIM),
            "configuration": configuration, "inference_counts": report["inference_counts"],
            "protected_files_unchanged": True, "rerun_allowed": False, "use_for_tuning_allowed": False,
            "report_file_sha256": {name: sha256(output / name) for name in
                                   ("report.md", "report.json", "predictions.csv", "inference_journal.jsonl")},
        }
        write_json_new(EVALUATION_LOCK, lock)
        # Protect the new evaluation evidence without modifying any original input or freeze manifest.
        for path in (*output.iterdir(), RUN_CLAIM, EVALUATION_LOCK):
            path.chmod(stat.S_IREAD)
        print(markdown, flush=True)
        print(f"Báo cáo: {output}\nKhóa đánh giá: {EVALUATION_LOCK}", flush=True)
    except Exception as error:
        write_json_new(output / "failure.json", {"status": "FAILED_DO_NOT_RERUN", "failed_at_utc": now(),
                       "error": repr(error), "note": "Keep claim intact. Read any saved journal; never repeat final-test inference."})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Chỉ dùng dữ liệu giả; không nạp model hoặc đọc final test.")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main()
