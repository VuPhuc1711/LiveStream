r"""Đánh giá cố định Rules, PhoBERT và Hybrid trên 30 câu test nội bộ.

Chạy bằng .venv-nlp/Scripts/python.exe -B -X utf8 evaluate_hybrid_test.py.
Chỉ dự đoán; không huấn luyện, đổi ngưỡng, sửa dữ liệu hoặc ghi checkpoint.
Kết quả được lưu trong một thư mục mới dưới reports/ sau mỗi lần chạy.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from nlp.hybrid_detector import HybridDetector, SAFE_THRESHOLD, WARNING_THRESHOLD
from nlp.phobert_detector import LABELS, PhoBertDetector


ROOT = Path(__file__).resolve().parent
# Historical v01 evaluation stays explicitly pinned, independent of the live default.
DEFAULT_CHECKPOINT = ROOT / "models/phobert_20260909_182641_815167/best"
METHODS = {"rules": "Rules", "phobert": "PhoBERT", "hybrid": "HybridDetector"}
RULE_MAPPING = {
    "match": "CAN_XAC_MINH",
    "no_match": "KHONG_CANH_BAO",
    "note": "Rules chỉ sàng lọc; không tự dự đoán CANH_BAO và không có confidence.",
}
ERROR_TYPES = {
    "false_review": "Đưa đi kiểm tra nhầm: KHONG_CANH_BAO → nhãn khác",
    "missed_review": "Bỏ sót sàng lọc: CAN_XAC_MINH/CANH_BAO → KHONG_CANH_BAO",
    "false_warning": "Cảnh báo mức cao nhầm: nhãn khác → CANH_BAO",
    "missed_warning": "Không giữ mức cảnh báo: CANH_BAO → nhãn khác",
    "warning_to_review": "Hạ xuống xác minh: CANH_BAO → CAN_XAC_MINH; vẫn được xem lại",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_key(text):
    return " ".join(unicodedata.normalize("NFC", text).split()).casefold()


def read_split(path, split):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"id", "text", "label", "family_id", "split", "review_status"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Thiếu cột trong {path}: {required - set(reader.fieldnames or [])}")
        columns = reader.fieldnames
        rows = list(reader)
    if not rows:
        raise ValueError(f"Tập {split} rỗng.")
    ids, texts = set(), set()
    for row in rows:
        if None in row or any(not isinstance(row.get(key), str) or not row[key].strip() for key in required):
            raise ValueError(f"Dòng CSV thiếu dữ liệu hoặc sai số cột trong {path}.")
        if row["label"] not in LABELS or row["split"] != split:
            raise ValueError(f"Sai nhãn/split tại {row['id']} trong {path}.")
        if row["id"] in ids or normalized_key(row["text"]) in texts:
            raise ValueError(f"Trùng ID/nội dung trong {path}: {row['id']}")
        ids.add(row["id"])
        texts.add(normalized_key(row["text"]))
    return rows, columns


def audit_test_split(test_rows):
    """Chỉ đọc train/validation để kiểm tra tách tập và đối chiếu lần huấn luyện."""
    metadata_path = DEFAULT_CHECKPOINT.parent / "run_info.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    audit = {}
    for split in ("train", "validation"):
        path = ROOT / "data" / "splits" / f"{split}.csv"
        rows, _ = read_split(path, split)
        overlaps = {
            key: sorted({r[key] for r in test_rows} & {r[key] for r in rows})
            for key in ("id", "family_id")
        }
        overlaps["normalized_text"] = sorted(
            {normalized_key(r["text"]) for r in test_rows}
            & {normalized_key(r["text"]) for r in rows}
        )
        training_ids = set(metadata["data_ids"][split])
        overlaps["recorded_training_ids"] = sorted({r["id"] for r in test_rows} & training_ids)
        if any(overlaps.values()):
            raise ValueError(f"Test giao với {split}: {overlaps}")
        matches_hash = sha256(path) == metadata["data_sha256"][split]
        matches_ids = {row["id"] for row in rows} == training_ids
        if not matches_hash or not matches_ids:
            raise ValueError(f"Tập {split} không khớp dữ liệu của checkpoint đã lưu.")
        audit[split] = {
            "overlap_counts": {key: len(value) for key, value in overlaps.items()},
            "matches_recorded_sha256": matches_hash,
            "matches_recorded_ids": matches_ids,
        }
    return {
        "comparisons": audit,
        "training_run_recorded_test_used": metadata.get("test_used"),
        "scope": "Tách khỏi train/validation theo ID, family_id và nội dung chuẩn hóa; không chứng minh độc lập ngoài thực tế.",
    }


def error_types(truth, predicted):
    tags = []
    if truth == "KHONG_CANH_BAO" and predicted != "KHONG_CANH_BAO":
        tags.append("false_review")
    if truth != "KHONG_CANH_BAO" and predicted == "KHONG_CANH_BAO":
        tags.append("missed_review")
    if truth != "CANH_BAO" and predicted == "CANH_BAO":
        tags.append("false_warning")
    if truth == "CANH_BAO" and predicted != "CANH_BAO":
        tags.append("missed_warning")
    if truth == "CANH_BAO" and predicted == "CAN_XAC_MINH":
        tags.append("warning_to_review")
    return tags


def evaluate_method(records, method):
    truth = [row["true_label"] for row in records]
    predictions = [row["predictions"][method] for row in records]
    report = classification_report(
        truth, predictions, labels=list(LABELS), output_dict=True, zero_division=0
    )
    per_label = {}
    for label in LABELS:
        values = report[label]
        per_label[label] = {
            "precision": values["precision"], "recall": values["recall"],
            "f1": values["f1-score"], "support": int(values["support"]),
            "false_positive_ids": [
                row["id"] for row in records
                if row["true_label"] != label and row["predictions"][method] == label
            ],
            "false_negative_ids": [
                row["id"] for row in records
                if row["true_label"] == label and row["predictions"][method] != label
            ],
        }
    errors = [
        {"id": row["id"], "text": row["text"], "true_label": row["true_label"],
         "predicted_label": row["predictions"][method],
         "error_types": error_types(row["true_label"], row["predictions"][method])}
        for row in records if row["true_label"] != row["predictions"][method]
    ]
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "correct": len(records) - len(errors), "total": len(records),
        "per_label": per_label,
        "confusion_matrix": confusion_matrix(truth, predictions, labels=list(LABELS)).tolist(),
        "matrix_label_order": list(LABELS), "matrix_rows": "true", "matrix_columns": "predicted",
        "errors": errors,
        "error_groups": {
            tag: [row["id"] for row in errors if tag in row["error_types"]] for tag in ERROR_TYPES
        },
    }


def markdown_cell(value):
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def render_report(report):
    lines = [
        "# Đánh giá Rules, PhoBERT và Hybrid trên tập test",
        "", f"Tập test: `{report['dataset']['path']}` — 30 câu.",
        "Phân bố nhãn: " + ", ".join(f"{label}: {count}" for label, count in report["dataset"]["label_counts"].items()),
        "", "Rules: khớp → CAN_XAC_MINH; không khớp → KHONG_CANH_BAO; không dự đoán CANH_BAO.",
        f"Hybrid giữ nguyên ngưỡng: CANH_BAO ≥ {WARNING_THRESHOLD:.0%}; KHONG_CANH_BAO ≥ {SAFE_THRESHOLD:.0%}.",
        "Không tinh chỉnh mô hình, Rules hoặc ngưỡng trên tập test.",
        "final_confidence là xác suất PhoBERT của nhãn cuối, chưa hiệu chỉnh cho hybrid.",
        "Dữ liệu mô phỏng nội bộ; cả 30 nhãn mang trạng thái DA_DUYET_AI, chưa phải đánh giá thực tế độc lập.",
        "", "## Chỉ số tổng hợp", "",
        "| Phương pháp | Đúng/Tổng | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, name in METHODS.items():
        result = report["methods"][method]
        lines.append(
            f"| {name} | {result['correct']}/{result['total']} | {result['accuracy']:.2%} | "
            f"{result['macro_precision']:.2%} | {result['macro_recall']:.2%} | {result['macro_f1']:.2%} |"
        )
    lines += ["", "Các chỉ số không xác định do không có dự đoán/mẫu được đặt bằng 0 (zero_division=0).",
              "", "## Quy ước lỗi", ""]
    lines.extend(f"- `{tag}`: {description}." for tag, description in ERROR_TYPES.items())
    lines += ["", "Một câu có thể thuộc nhiều nhóm lỗi; không cộng các nhóm để tính tổng câu sai."]
    for method, name in METHODS.items():
        result = report["methods"][method]
        lines += ["", f"## {name}", "", "| Nhãn | Precision | Recall | F1 | Số câu thật |",
                  "|---|---:|---:|---:|---:|"]
        for label, values in result["per_label"].items():
            lines.append(
                f"| {label} | {values['precision']:.2%} | {values['recall']:.2%} | "
                f"{values['f1']:.2%} | {values['support']} |"
            )
        lines += ["", "Confusion matrix: hàng = nhãn thật, cột = nhãn dự đoán.", "",
                  "| Thật \\ Dự đoán | " + " | ".join(LABELS) + " |", "|---|---:|---:|---:|"]
        for label, counts in zip(LABELS, result["confusion_matrix"]):
            lines.append(f"| {label} | " + " | ".join(map(str, counts)) + " |")
        lines += ["", "Nhầm và bỏ sót theo từng nhãn:", "",
                  "| Nhãn | Dự đoán nhầm nhãn này (FP): ID | Bỏ sót nhãn này (FN): ID |", "|---|---|---|"]
        for label, values in result["per_label"].items():
            lines.append(
                f"| {label} | {', '.join(values['false_positive_ids']) or 'Không'} | "
                f"{', '.join(values['false_negative_ids']) or 'Không'} |"
            )
        lines += ["", f"Toàn bộ {len(result['errors'])} câu dự đoán sai:", "",
                  "| ID | Câu | Nhãn thật | Dự đoán | Loại lỗi |", "|---|---|---|---|---|"]
        for error in result["errors"]:
            lines.append("| " + " | ".join(markdown_cell(value) for value in (
                error["id"], error["text"], error["true_label"], error["predicted_label"],
                ", ".join(error["error_types"]),
            )) + " |")
        for tag, ids in result["error_groups"].items():
            lines += ["", f"- {ERROR_TYPES[tag]}: **{len(ids)}** câu — {', '.join(ids) or 'Không'}."]
    lines += ["", "Dữ liệu, checkpoint và mã nguồn được bảo vệ có SHA-256 không đổi trước/sau đánh giá.", ""]
    return "\n".join(lines)


def save_predictions(path, records):
    columns = ["id", "text", "true_label", "method", "predicted_label", "confidence", "correct",
               "error_types", "matched_keywords", *[f"p_{label}" for label in LABELS], "explanation"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in records:
            hybrid = row["hybrid_result"]
            phobert = hybrid["phobert_result"]
            keywords = list(dict.fromkeys(item["keyword"] for item in hybrid["rule_result"]["findings"]))
            for method in METHODS:
                predicted = row["predictions"][method]
                confidence = {"rules": "", "phobert": phobert["confidence"], "hybrid": hybrid["final_confidence"]}[method]
                explanation = {
                    "rules": "Khớp Rules → CAN_XAC_MINH; không khớp → KHONG_CANH_BAO.",
                    "phobert": "Nhãn có xác suất softmax cao nhất của checkpoint PhoBERT.",
                    "hybrid": hybrid["explanation"],
                }[method]
                writer.writerow({
                    "id": row["id"], "text": row["text"], "true_label": row["true_label"],
                    "method": method, "predicted_label": predicted, "confidence": confidence,
                    "correct": predicted == row["true_label"],
                    "error_types": "; ".join(error_types(row["true_label"], predicted)),
                    "matched_keywords": "; ".join(keywords), "explanation": explanation,
                    **{f"p_{label}": "" if method == "rules" else phobert["probabilities"][label] for label in LABELS},
                })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args()
    test_path = ROOT / "data" / "splits" / "test.csv"
    protected_paths = sorted(set(
        [path for path in (ROOT / "data").rglob("*") if path.is_file()]
        + [path for path in DEFAULT_CHECKPOINT.parent.rglob("*") if path.is_file()]
        + list((ROOT / "nlp").glob("*.py"))
        + [ROOT / name for name in ("main.py", "train_phobert.py", "prepare_dataset.py",
                                    "evaluate_keywords.py", "test_phobert.py", "test_hybrid.py")]
    ))
    print("Đang ghi nhận SHA-256 và kiểm tra tập test...", flush=True)
    before = {str(path.relative_to(ROOT)): sha256(path) for path in protected_paths}
    rows, columns = read_split(test_path, "test")
    if len(rows) != 30 or set(row["label"] for row in rows) != set(LABELS):
        raise ValueError("Cần đúng 30 câu test với đủ ba nhãn; dừng để kiểm tra dữ liệu.")
    audit = audit_test_split(rows)
    print("Đã xác nhận test không giao ID, family_id, nội dung với train/validation.", flush=True)
    phobert = PhoBertDetector(checkpoint_path=DEFAULT_CHECKPOINT, device=args.device)
    detector = HybridDetector(phobert_detector=phobert)
    records = []
    for index, row in enumerate(rows, start=1):
        # Chỉ text đi vào detector; nhãn/metadata chỉ dùng chấm điểm sau dự đoán.
        result = detector.predict(row["text"])
        records.append({
            "id": row["id"], "text": row["text"], "true_label": row["label"],
            "family_id": row["family_id"],
            "predictions": {
                "rules": RULE_MAPPING["match" if result["rule_result"]["needs_review"] else "no_match"],
                "phobert": result["phobert_result"]["label"], "hybrid": result["final_label"],
            },
            "hybrid_result": result,
        })
        print(f"Đã dự đoán {index}/{len(rows)}: {row['id']}", flush=True)
    after = {str(path.relative_to(ROOT)): sha256(path) for path in protected_paths}
    if before != after:
        raise RuntimeError("Có file được bảo vệ thay đổi trong lúc đánh giá; không ghi báo cáo hoàn tất.")
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"path": str(test_path), "sha256": sha256(test_path), "row_count": len(rows),
                    "columns": columns, "label_counts": dict(Counter(row["label"] for row in rows)),
                    "review_status_counts": dict(Counter(row["review_status"] for row in rows)),
                    "family_count": len({row["family_id"] for row in rows})},
        "split_audit": audit, "checkpoint": str(DEFAULT_CHECKPOINT), "device": str(phobert.device),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "underthesea", "numpy", "scikit-learn")},
        "rules_mapping": RULE_MAPPING,
        "hybrid_policy": {"warning_threshold": WARNING_THRESHOLD, "safe_threshold": SAFE_THRESHOLD,
                          "fallback": "CAN_XAC_MINH", "thresholds_tuned_on_test": False,
                          "final_confidence": "PhoBERT probability of final_label; not calibrated hybrid confidence"},
        "integrity": {"unchanged": before == after, "protected_file_sha256": before,
                      "evaluator_sha256": sha256(Path(__file__).resolve())},
        "metric_policy": {"labels": list(LABELS), "zero_division": 0},
        "error_definitions": ERROR_TYPES,
        "limitations": ["Small synthetic internal test set; all 30 labels are AI-reviewed.",
                        "Rules baseline uses a declared mapping from binary screening to three labels.",
                        "Existing evaluation script targets the full dataset; historical test exposure while designing Rules is unknown.",
                        "No training, threshold search or model selection is performed in this evaluation."],
        "methods": {method: evaluate_method(records, method) for method in METHODS},
        "predictions": records,
    }
    output_dir = ROOT / "reports" / ("hybrid_test_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    save_predictions(output_dir / "predictions.csv", records)
    summary = render_report(report)
    (output_dir / "report.md").write_text(summary, encoding="utf-8")
    print("\n" + summary, flush=True)
    print(f"Đã lưu JSON, CSV và báo cáo dễ đọc tại: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
