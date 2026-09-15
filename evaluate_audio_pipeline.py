"""Đối chiếu JSON Whisper/demo với 9 câu thu âm; không chạy lại Whisper hoặc train.

Ví dụ: python evaluate_audio_pipeline.py reports/audio_demo_<timestamp>/result.json
Dùng --alignment-only để xem phép ghép mà không nạp PhoBERT.
"""

import argparse
import csv
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

from main import CHECKPOINT, ROOT, PROTECTED_CODE, sha256, utc_now, verify_checkpoint, write_json
from speech.sentence_units import build_classification_units


GROUND_TRUTH = ROOT / "data/audio_test_ground_truth.csv"
LABELS = ("KHONG_CANH_BAO", "CAN_XAC_MINH", "CANH_BAO")
# Chỉ là ngưỡng sàng lọc phép ghép để nghe lại, hoàn toàn độc lập với ngưỡng Hybrid.
ALIGNMENT_MIN_SIMILARITY = 0.60
MAX_HYPOTHESIS_TOKENS = 4000


def tokens(text):
    return [match.group().casefold() for match in re.finditer(r"[^\W_]+", unicodedata.normalize("NFC", text))]


def normalized_text(text):
    return " ".join(tokens(text))


def load_ground_truth(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"id", "text", "true_label", "reason"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV ground truth thiếu cột id,text,true_label,reason.")
        rows = list(reader)
    if len(rows) != 9 or any(None in row or any(not isinstance(row.get(k), str) or not row[k].strip() for k in required) for row in rows):
        raise ValueError("Ground truth phải có đúng 9 dòng đầy đủ dữ liệu.")
    if set(row["true_label"] for row in rows) != set(LABELS):
        raise ValueError("Ground truth cần đủ ba nhãn hợp lệ; không ép cân bằng sau duyệt.")
    if len({row["id"] for row in rows}) != 9 or len({normalized_text(row["text"]) for row in rows}) != 9:
        raise ValueError("Ground truth bị trùng ID hoặc text chuẩn hóa.")
    if any(not tokens(row["text"]) for row in rows):
        raise ValueError("Câu gốc không có từ để đối chiếu.")
    return rows


def audit_ground_truth(path):
    rows = load_ground_truth(path)
    comparisons = {}
    for relative in ("data/dataset_livestream_v02.csv", "data/final_test_v03_chua_duyet.csv"):
        source = ROOT / relative
        before = sha256(source)
        with source.open(encoding="utf-8-sig", newline="") as stream:
            others = list(csv.DictReader(stream))
        overlaps = [{"id": row["id"], "existing_id": other["id"]}
                    for row in rows for other in others
                    if normalized_text(row["text"]) == normalized_text(other["text"])]
        if overlaps:
            raise ValueError(f"Trùng text với {relative}: {overlaps}")
        if sha256(source) != before:
            raise RuntimeError(f"Nguồn thay đổi khi kiểm tra: {source}")
        comparisons[relative] = {"rows": len(others), "duplicate_normalized_text": overlaps, "sha256": before}
    return {"status": "VALID", "ground_truth": str(path.resolve()), "sha256": sha256(path),
            "rows": 9, "label_counts": dict(Counter(row["true_label"] for row in rows)),
            "unique_ids": 9, "unique_normalized_texts": 9, "comparisons": comparisons,
            "normalization": "NFC, casefold, punctuation to token boundaries, collapse whitespace; keep Vietnamese accents and digits",
            "label_provenance": "Theo metadata duyệt của CSV; bản cũ không có metadata thì chưa xác nhận người duyệt.",
            "review_status_counts": dict(Counter(row.get("review_status", "UNSPECIFIED") for row in rows)),
            "reviewers": sorted({row.get("reviewer", "UNSPECIFIED") for row in rows}),
            "model_inference_performed": False}


def edit_alignment(reference, hypothesis):
    """Levenshtein toàn cục theo thứ tự; không dùng nhãn hoặc dự đoán để chọn phép ghép."""
    n, m = len(reference), len(hypothesis)
    if n > 4000 or m > MAX_HYPOTHESIS_TOKENS:
        raise ValueError("Transcript quá dài cho bản thu 9 câu; hãy kiểm tra có chọn nhầm JSON không.")
    costs = [list(range(m + 1))]
    for i in range(1, n + 1):
        current = [i] + [0] * m
        previous = costs[-1]
        for j in range(1, m + 1):
            current[j] = min(previous[j - 1] + (reference[i - 1] != hypothesis[j - 1]),
                             previous[j] + 1, current[j - 1] + 1)
        costs.append(current)
    operations = []
    i, j = n, m
    while i or j:
        if i and j and costs[i][j] == costs[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]):
            operations.append((i - 1, j - 1, "equal" if reference[i - 1] == hypothesis[j - 1] else "substitute"))
            i, j = i - 1, j - 1
        elif i and costs[i][j] == costs[i - 1][j] + 1:
            operations.append((i - 1, None, "delete"))
            i -= 1
        else:
            operations.append((None, j - 1, "insert"))
            j -= 1
    return list(reversed(operations))


def text_metrics(reference_text, hypothesis_text):
    reference, hypothesis = tokens(reference_text), tokens(hypothesis_text)
    operations = edit_alignment(reference, hypothesis)
    counts = Counter(op for _, _, op in operations)
    distance = sum(counts[op] for op in ("substitute", "delete", "insert"))
    return {"normalized_exact_match": reference == hypothesis,
            "word_similarity": 1 - distance / max(len(reference), len(hypothesis), 1),
            "character_similarity": SequenceMatcher(None, " ".join(reference), " ".join(hypothesis), autojunk=False).ratio(),
            "wer": distance / len(reference) if reference else None,
            "reference_tokens": len(reference), "hypothesis_tokens": len(hypothesis),
            "substitutions": counts["substitute"], "deletions": counts["delete"], "insertions": counts["insert"],
            "edits": [{"operation": op, "reference": reference[i] if i is not None else None,
                       "hypothesis": hypothesis[j] if j is not None else None}
                      for i, j, op in operations if op != "equal"]}


def validate_prediction(value):
    if value["final_label"] not in LABELS or value["phobert_result"]["label"] not in LABELS:
        raise ValueError("Nhãn Hybrid/PhoBERT trong JSON không hợp lệ.")
    pho = value["phobert_result"]
    probs = pho["probabilities"]
    if set(probs) != set(LABELS) or not all(math.isfinite(p) and 0 <= p <= 1 for p in probs.values()):
        raise ValueError("Xác suất trong JSON không hợp lệ.")
    if (not math.isclose(sum(probs.values()), 1, abs_tol=1e-5)
            or pho["confidence"] != probs[pho["label"]]
            or value["final_confidence"] != probs[value["final_label"]]):
        raise ValueError("Confidence trong JSON không khớp xác suất PhoBERT.")
    expected = ("CANH_BAO" if pho["label"] == "CANH_BAO" and pho["confidence"] >= 0.55
                else "KHONG_CANH_BAO" if pho["label"] == "KHONG_CANH_BAO" and pho["confidence"] >= 0.65
                else "CAN_XAC_MINH")
    if value["final_label"] != expected or not isinstance(value["rule_result"]["findings"], list):
        raise ValueError("Kết quả không đúng chính sách Hybrid đang giữ nguyên.")
    return {key: value[key] for key in ("final_label", "final_confidence", "phobert_result", "rule_result", "explanation")}


def validate_classification_units(units, raw_segments):
    """Kiểm tra truy vết/thời gian; không suy ra ranh giới câu từ ground truth."""
    seen = set()
    source_counts = Counter(unit["source_segment_id"] for unit in units)
    for unit in units:
        if not unit["unit_id"] or unit["unit_id"] in seen or not unit["text"].strip():
            raise ValueError("Classification unit bị rỗng hoặc trùng unit_id.")
        seen.add(unit["unit_id"])
        parents = [s for index, s in enumerate(raw_segments) if s.get("id", index) == unit["source_segment_id"]]
        if len(parents) != 1:
            raise ValueError("Classification unit không có source_segment_id duy nhất trong whisper_segments.")
        parent = parents[0]
        if not (math.isfinite(unit["start"]) and math.isfinite(unit["end"])
                and parent["start"] <= unit["start"] <= unit["end"] <= parent["end"]):
            raise ValueError("Timestamp unit nằm ngoài segment nguồn.")
        if type(unit["timestamp_is_approximate"]) is not bool:
            raise ValueError("timestamp_is_approximate phải là bool.")
        if source_counts[unit["source_segment_id"]] > 1 and not unit["timestamp_is_approximate"]:
            raise ValueError("Các câu tách từ cùng segment phải đánh dấu timestamp ước lượng.")
        if "source_text_start" in unit and parent["text"][unit["source_text_start"]:unit["source_text_end"]] != unit["text"]:
            raise ValueError("Text unit không khớp đoạn ký tự trong raw Whisper segment.")


def load_source(path):
    with path.open(encoding="utf-8") as stream:
        source = json.load(stream)
    raw_segments = source.get("whisper_segments", source.get("segments"))
    if not isinstance(raw_segments, list):
        raise ValueError("Chọn whisper.json hoặc result.json có danh sách segments do main.py xuất.")
    if "classification_units" in source:
        if not isinstance(source["classification_units"], list):
            raise ValueError("classification_units phải là danh sách, không fallback sang raw segment.")
        segments = [dict(unit, id=unit["unit_id"]) for unit in source["classification_units"]]
        validate_classification_units(segments, raw_segments)
        unit_kind = "classification_units"
    elif "checkpoint" not in source:
        # whisper.json chưa có nhãn: cũng tách bằng đúng thuật toán của main trước khi phân loại.
        segments = build_classification_units(raw_segments)
        unit_kind = "classification_units"
    else:
        # Báo cáo cũ được chấm đúng các dự đoán cũ, không âm thầm thay bằng pipeline mới.
        segments = source["segments"]
        unit_kind = "whisper_segments_legacy"
    source = {**source, "_evaluation_unit_kind": unit_kind, "_whisper_segment_count": len(raw_segments)}
    predictions = None
    if "checkpoint" in source:
        if source.get("status") not in ("COMPLETED", "NO_SPEECH"):
            raise ValueError("Lần chạy audio chưa hoàn tất.")
        # Saved v02 predictions remain v02 even after the live default changes.
        checkpoint = verify_checkpoint(source["checkpoint"]["actual_loaded_path"])
        if source["checkpoint"]["file_sha256"] != checkpoint["file_sha256"]:
            raise ValueError("Kết quả audio không khớp checkpoint đã khóa của chính báo cáo.")
        if source.get("hybrid_thresholds") != {"CANH_BAO": 0.55, "KHONG_CANH_BAO": 0.65}:
            raise ValueError("Kết quả audio sử dụng ngưỡng khác.")
        # Changing the default checkpoint path does not invalidate cached
        # predictions. Rules and decision thresholds must still match.
        policy_files = ("nlp/keyword_detector.py", "nlp/hybrid_detector.py")
        if any(source.get("protected_code_sha256", {}).get(name) != sha256(ROOT / name) for name in policy_files):
            raise ValueError("Rules/Hybrid hiện tại khác chính sách dùng để tạo JSON audio.")
        predictions = [validate_prediction(s) if s.get("status") == "CLASSIFIED" else None for s in segments]
    previous_start = -1.0
    for segment in segments:
        start, end = segment["start"], segment["end"]
        if (not isinstance(segment["text"], str) or not math.isfinite(start) or not math.isfinite(end)
                or not 0 <= start <= end or start < previous_start):
            raise ValueError("Segment thiếu text hoặc có mốc thời gian/thứ tự không hợp lệ.")
        previous_start = start
    return segments, predictions, source


def anchored_operations(reference, hypothesis, spans):
    """Neo câu trùng token duy nhất theo thứ tự, rồi căn chỉnh các khoảng còn lại.

Tránh lấy một từ lặp ở cuối câu nguyên vẹn để gán sang câu kế tiếp bị thiếu.
Nếu các neo xung đột thứ tự/chéo nhau, bỏ neo và dùng phép ghép toàn cục.
"""
    anchors = []
    for begin, end in spans:
        phrase = reference[begin:end]
        matches = [index for index in range(len(hypothesis) - len(phrase) + 1)
                   if hypothesis[index:index + len(phrase)] == phrase]
        if len(matches) == 1:
            anchors.append((begin, end, matches[0], matches[0] + len(phrase)))
    if any(anchors[index][2] < anchors[index - 1][3] for index in range(1, len(anchors))):
        return edit_alignment(reference, hypothesis)
    operations = []
    ref_start, hyp_start = 0, 0
    for ref_begin, ref_end, hyp_begin, hyp_end in anchors + [(len(reference), len(reference), len(hypothesis), len(hypothesis))]:
        for ref, hyp, operation in edit_alignment(reference[ref_start:ref_begin], hypothesis[hyp_start:hyp_begin]):
            operations.append((ref_start + ref if ref is not None else None,
                               hyp_start + hyp if hyp is not None else None, operation))
        operations.extend((ref_begin + index, hyp_begin + index, "equal") for index in range(ref_end - ref_begin))
        ref_start, hyp_start = ref_end, hyp_end
    return operations


def align_sentences(rows, segments):
    reference, ref_owners, spans = [], [], []
    for index, row in enumerate(rows):
        words = tokens(row["text"])
        spans.append((len(reference), len(reference) + len(words)))
        reference.extend(words)
        ref_owners.extend([index] * len(words))
    hypothesis, positions = [], []
    for index, segment in enumerate(segments):
        text = unicodedata.normalize("NFC", segment["text"])
        for match in re.finditer(r"[^\W_]+", text):
            hypothesis.append(match.group().casefold())
            positions.append({"segment": index, "begin": match.start(), "end": match.end()})
    if len(hypothesis) > MAX_HYPOTHESIS_TOKENS:
        raise ValueError("Transcript quá dài cho bản thu 9 câu; hãy kiểm tra JSON đầu vào.")
    operations = anchored_operations(reference, hypothesis, spans)
    owners, boundary_insertions = {}, set()
    last_ref = None
    for offset, (ref, hyp, operation) in enumerate(operations):
        if ref is not None:
            last_ref = ref
        if hyp is None:
            continue
        if ref is not None:
            owners[hyp] = ref_owners[ref]
        else:
            next_ref = next((r for r, _, _ in operations[offset + 1:] if r is not None), None)
            owner = ref_owners[last_ref] if last_ref is not None else ref_owners[next_ref] if next_ref is not None else 0
            owners[hyp] = owner
            if next_ref is not None and owner != ref_owners[next_ref]:
                boundary_insertions.update((owner, ref_owners[next_ref]))
    segment_owners = [set() for _ in segments]
    for hyp, owner in owners.items():
        segment_owners[positions[hyp]["segment"]].add(owner)
    global_metrics = text_metrics(" ".join(row["text"] for row in rows), " ".join(s["text"] for s in segments))
    global_reliable = global_metrics["word_similarity"] >= ALIGNMENT_MIN_SIMILARITY
    aligned = []
    for index, row in enumerate(rows):
        selected = [h for h in range(len(hypothesis)) if owners[h] == index]
        groups = []
        for h in selected:
            if groups and h == groups[-1][-1] + 1 and positions[h]["segment"] == positions[groups[-1][-1]]["segment"]:
                groups[-1].append(h)
            else:
                groups.append([h])
        parts = []
        for group in groups:
            first, last = positions[group[0]], positions[group[-1]]
            segment_index = first["segment"]
            segment = segments[segment_index]
            text = unicodedata.normalize("NFC", segment["text"])
            shared = len(segment_owners[segment_index]) > 1
            parts.append({"segment_index": segment_index, "segment_id": segment.get("id", segment_index),
                          "unit_id": segment.get("unit_id"), "source_segment_id": segment.get("source_segment_id", segment.get("id", segment_index)),
                          "timestamp_is_approximate": segment.get("timestamp_is_approximate", False),
                          "start": segment["start"], "end": segment["end"], "shared_segment": shared,
                          "text": text[first["begin"]:last["end"]] if shared else text})
        transcript = " ".join(part["text"].strip() for part in parts)
        metrics = text_metrics(row["text"], transcript)
        reliable = bool(selected) and global_reliable and metrics["word_similarity"] >= ALIGNMENT_MIN_SIMILARITY and index not in boundary_insertions
        aligned.append({**row, "transcript": transcript, "segment_parts": parts,
                        "segment_indexes": sorted({part["segment_index"] for part in parts}),
                        "start": min((part["start"] for part in parts), default=None),
                        "end": max((part["end"] for part in parts), default=None),
                        "unit_ids": list(dict.fromkeys(part["unit_id"] for part in parts if part["unit_id"] is not None)),
                        "source_segment_ids": list(dict.fromkeys(part["source_segment_id"] for part in parts)),
                        "timestamp_is_approximate": any(part["timestamp_is_approximate"] or part["shared_segment"] for part in parts),
                        "text_metrics": metrics,
                        "alignment_status": "MISSING" if not selected else "ALIGNED_CANDIDATE" if reliable else "NEEDS_REVIEW",
                        "alignment_eligible_for_scoring": reliable,
                        "has_shared_segment": any(part["shared_segment"] for part in parts),
                        "asr_text_status": "MATCH_NORMALIZED" if metrics["normalized_exact_match"] else "DIFFERENT_NEEDS_LISTEN",
                        "boundary_insertion_needs_review": index in boundary_insertions})
    return {"sentences": aligned, "segment_owners": [sorted(owners) for owners in segment_owners],
            "global_text_metrics": global_metrics, "global_alignment_eligible": global_reliable}


def classification_metrics(pairs, expected_count):
    matrix = [[0] * 3 for _ in LABELS]
    for truth, prediction in pairs:
        matrix[LABELS.index(truth)][LABELS.index(prediction)] += 1
    total, correct = len(pairs), sum(matrix[i][i] for i in range(3))
    per_label = {}
    for i, label in enumerate(LABELS):
        support, predicted = sum(matrix[i]), sum(matrix[j][i] for j in range(3))
        precision, recall = matrix[i][i] / predicted if predicted else 0, matrix[i][i] / support if support else 0
        per_label[label] = {"precision": precision, "recall": recall,
                            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0, "support": support}
    return {"scored": total, "expected": expected_count, "coverage": total / expected_count if expected_count else None,
            "correct": correct, "accuracy": correct / total if total else None,
            "macro_f1": sum(s["f1"] for s in per_label.values()) / 3 if total else None,
            "per_label": per_label, "confusion_matrix": matrix, "label_order": list(LABELS),
            "matrix_rows": "true", "matrix_columns": "predicted", "zero_division": 0}


def score_alignment(alignment, segments, segment_predictions, predict=None, unit_kind="whisper_segments_legacy"):
    """Chấm đơn vị thực tế đã được chọn; câu tái ghép chỉ dùng để chẩn đoán."""
    clean_pairs, reconstructed_pairs, pipeline_pairs, segment_results = [], [], [], []
    sentences = alignment["sentences"]
    for index, (segment, owners) in enumerate(zip(segments, alignment["segment_owners"])):
        prediction = segment_predictions[index] if segment_predictions is not None else None
        eligible = len(owners) == 1 and sentences[owners[0]]["alignment_eligible_for_scoring"] and prediction is not None
        truth = sentences[owners[0]]["true_label"] if eligible else None
        if eligible:
            pipeline_pairs.append((truth, prediction["final_label"]))
        segment_results.append({"segment_index": index, "id": segment.get("id", index),
                               "unit_id": segment.get("unit_id"), "source_segment_id": segment.get("source_segment_id", segment.get("id", index)),
                               "timestamp_is_approximate": segment.get("timestamp_is_approximate", False),
                               "start": segment["start"], "end": segment["end"], "text": segment["text"],
                               "candidate_reference_ids": [sentences[owner]["id"] for owner in owners],
                               "eligible_for_scoring": eligible, "true_label": truth, "prediction": prediction,
                               "correct": truth == prediction["final_label"] if eligible else None,
                               "excluded_reason": None if eligible else "Shared, missing, low-similarity alignment or no saved prediction"})
    for sentence in sentences:
        clean = predict(sentence["text"]) if predict is not None else None
        reconstructed = predict(sentence["transcript"]) if predict is not None and sentence["alignment_eligible_for_scoring"] else None
        if clean is not None:
            clean_pairs.append((sentence["true_label"], clean["final_label"]))
        if reconstructed is not None:
            reconstructed_pairs.append((sentence["true_label"], reconstructed["final_label"]))
        actual = [segment_results[index] for index in sentence["segment_indexes"]]
        all_segments_scored = bool(actual) and all(item["eligible_for_scoring"] for item in actual)
        pipeline_correct = all(item["correct"] for item in actual) if all_segments_scored else None
        clean_error = clean["final_label"] != sentence["true_label"] if clean else None
        reconstructed_error = reconstructed["final_label"] != sentence["true_label"] if reconstructed else None
        if not sentence["alignment_eligible_for_scoring"]:
            diagnosis = "Phép ghép thiếu hoặc chưa đáng tin; cần nghe lại, chưa quy lỗi phân loại từ audio."
        elif clean_error:
            diagnosis = "Hybrid đã sai trên câu gốc: lỗi phân loại tồn tại ngay cả khi chưa có ASR."
        elif reconstructed_error and not sentence["text_metrics"]["normalized_exact_match"]:
            diagnosis = "Câu gốc được phân loại đúng nhưng câu nhận dạng sau ghép bị sai; nghi ảnh hưởng ASR/phép ghép, cần nghe lại."
        elif pipeline_correct is False and clean_error is False:
            diagnosis = "Hybrid đúng trên câu gốc nhưng sai ở đơn vị phân loại thực tế; cần xem ranh giới câu và lời nhận dạng."
        elif sentence["has_shared_segment"]:
            diagnosis = "Một đơn vị phân loại chứa nhiều câu gốc; không sao chép nhãn cho từng câu. Kết quả câu tái ghép chỉ để chẩn đoán."
        elif clean is None:
            diagnosis = "Chưa chạy đối chứng trên câu gốc; chưa thể tách nguyên nhân lỗi ASR và lỗi phân loại."
        else:
            diagnosis = "Xem riêng độ khác transcript và nhãn; chữ khác chưa đủ chứng minh nhãn bị sai do Whisper."
        sentence.update({"clean_text_prediction": clean, "reconstructed_text_prediction": reconstructed,
                         "pipeline_segment_labels": [item["prediction"]["final_label"] if item["prediction"] else None for item in actual],
                         "pipeline_all_segments_correct": pipeline_correct,
                         "clean_text_classification_error": clean_error,
                         "reconstructed_text_classification_error": reconstructed_error, "diagnosis": diagnosis})
    scored_sentences = [s for s in sentences if s["pipeline_all_segments_correct"] is not None]
    strict_correct = sum(s["pipeline_all_segments_correct"] for s in scored_sentences)
    pipeline_key = "pipeline_classification_units" if unit_kind == "classification_units" else "pipeline_original_segments"
    return {"sentences": sentences, "pipeline_segments": segment_results, "pipeline_units": segment_results,
            "unit_kind": unit_kind, "pipeline_metric_key": pipeline_key,
            "alignment_summary": {"expected_sentence_count": len(sentences), "classification_unit_count": len(segments),
                "matched_reference_count": sum(bool(s["segment_indexes"]) for s in sentences),
                "missing_reference_ids": [s["id"] for s in sentences if not s["segment_indexes"]],
                "unscorable_reference_ids": [s["id"] for s in sentences if s["pipeline_all_segments_correct"] is None],
                "shared_unit_ids": [r["id"] for r in segment_results if len(r["candidate_reference_ids"]) > 1],
                "unmatched_unit_ids": [r["id"] for r in segment_results if not r["candidate_reference_ids"]]},
            "metrics": {pipeline_key: classification_metrics(pipeline_pairs, len(segments)),
                        "clean_reference_text_diagnostic": classification_metrics(clean_pairs, len(sentences)),
                        "reconstructed_asr_text_diagnostic": classification_metrics(reconstructed_pairs, len(sentences)),
                        "pipeline_sentence_strict": {"scored": len(scored_sentences), "expected": len(sentences),
                            "correct": strict_correct, "correct_out_of_all_nine": strict_correct / len(sentences),
                            "accuracy_on_scored_sentences": strict_correct / len(scored_sentences) if scored_sentences else None,
                            "note": "Một câu đúng khi mọi đơn vị riêng của câu đều đúng; câu thiếu/ghép chung/chưa chắc không được tính đúng. Không tổng hợp nhãn mới cho Hybrid."}},
            "error_lists": {"whisper_text_differences_need_listening": [s["id"] for s in sentences if not s["text_metrics"]["normalized_exact_match"]],
                            "alignment_needs_review": [s["id"] for s in sentences if not s["alignment_eligible_for_scoring"] or s["has_shared_segment"]],
                            "clean_text_classification_errors": [s["id"] for s in sentences if s["clean_text_classification_error"] is True],
                            "reconstructed_text_classification_errors": [s["id"] for s in sentences if s["reconstructed_text_classification_error"] is True],
                            "pipeline_sentence_errors_on_scored_only": [s["id"] for s in sentences if s["pipeline_all_segments_correct"] is False]}}


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def percent(value):
    return "chưa chấm" if value is None else f"{value:.2%}"


def render_report(report):
    summary = report["alignment_summary"]
    lines = ["# Kiểm thử có nhãn cho pipeline audio", "", f"Nguồn JSON: `{report['source_path']}`.",
             f"Ground truth: `{report['ground_truth_path']}`; SHA-256 `{report['ground_truth_sha256']}`.",
             f"Chế độ: {report['mode']}. Checkpoint: `{CHECKPOINT}`.", "",
             "Ghép theo thứ tự token: neo câu trùng nguyên văn sau chuẩn hóa nếu vị trí duy nhất, rồi căn chỉnh Levenshtein các khoảng còn lại. Không dùng nhãn để ghép.",
             "Chuẩn hóa NFC, chữ thường và dấu câu; giữ dấu tiếng Việt.",
             "WER dùng token cách nhau bằng khoảng trắng (tiếng Việt có thể là âm tiết). WER có thể lớn hơn 100% nếu thêm nhiều từ.",
             "Số viết bằng chữ/số có thể khác nhau dù đọc cùng nghĩa; không tự chuyển đổi số hoặc bỏ các từ phủ định.",
             "Khác chữ là tín hiệu cần nghe lại, có thể do Whisper, cách đọc hoặc phép ghép; chưa phải lỗi ASR đã được người nghe xác nhận.",
             "Mốc thời gian unit tách từ một segment được ước lượng theo độ dài, không phải word timestamp; xem timestamp_is_approximate.", "",
             "## Nhận dạng và phép ghép", "",
             f"Câu mong đợi: {summary['expected_sentence_count']}; raw Whisper segments: {summary['whisper_segment_count']}; đơn vị phân loại: {summary['classification_unit_count']}.",
             f"Nguồn ưu tiên để chấm: {report['unit_kind']}. Câu có nội dung ghép: {summary['matched_reference_count']}/{summary['expected_sentence_count']}.",
             f"Câu không ghép được: {', '.join(summary['missing_reference_ids']) or 'không có'}. Câu chưa chấm riêng được: {', '.join(summary['unscorable_reference_ids']) or 'không có'}.",
             f"Đơn vị dùng chung nhiều câu: {', '.join(map(str, summary['shared_unit_ids'])) or 'không có'}; đơn vị không ghép được: {', '.join(map(str, summary['unmatched_unit_ids'])) or 'không có'}.", "",
             f"WER toàn bản thu: {percent(report['alignment']['global_text_metrics']['wer'])}.",
             f"Độ giống token toàn bản thu: {percent(report['alignment']['global_text_metrics']['word_similarity'])}.",
             f"Ngưỡng xem lại phép ghép: {ALIGNMENT_MIN_SIMILARITY:.0%}; không phải ngưỡng quyết định của Hybrid.", "",
             "| ID | Giống token | WER | Phép ghép | Unit / chỉ số cũ | Nhãn thật | Nhãn đơn vị thực tế |",
             "|---|---:|---:|---|---|---|---|"]
    for row in report["sentences"]:
        lines.append(f"| {row['id']} | {percent(row['text_metrics']['word_similarity'])} | {percent(row['text_metrics']['wer'])} | "
                     f"{row['alignment_status']} | {cell(row['unit_ids'] or row['segment_indexes'])} | {row['true_label']} | {cell(row['pipeline_segment_labels'])} |")
    lines += ["", "## Phân loại — tách pipeline thực tế và chẩn đoán", "",
              "Ưu tiên classification_units và nhãn đã lưu; báo cáo cũ vẫn chấm segments cũ để so sánh. Với whisper.json, tách theo dấu câu trước khi phân loại.",
              "Câu ghép: tận dụng câu tham chiếu để khôi phục ranh giới, nên chỉ là chẩn đoán; không thay thế điểm pipeline thực tế.",
              "Câu gốc sạch: kiểm tra lỗi phân loại không qua Whisper. Cả hai chẩn đoán dùng cùng checkpoint/Rules/ngưỡng.", "",
              "| Phép đo | Đã chấm/Tổng | Accuracy | Macro-F1 |", "|---|---:|---:|---:|"]
    for name, metric in report["metrics"].items():
        if name == "pipeline_sentence_strict":
            continue
        lines.append(f"| {name} | {metric['scored']}/{metric['expected']} | {percent(metric['accuracy'])} | {percent(metric['macro_f1'])} |")
    strict = report["metrics"]["pipeline_sentence_strict"]
    strict_summary = (f"Câu đạt yêu cầu mọi đơn vị đều đúng: {strict['correct']}/9; có thể chấm theo phép ghép: {strict['scored']}/9."
                      if strict["scored"] else "Chưa có câu đủ điều kiện chấm pipeline; không diễn giải thành Accuracy bằng 0%.")
    lines += ["", strict_summary,
              "Các mẫu loại khỏi chấm vẫn được ghi trong báo cáo; không coi mất câu hoặc ghép chung là dự đoán đúng.", "",
              "## Lỗi nhận dạng và lỗi phân loại", ""]
    for name, ids in report["error_lists"].items():
        score_key = {"clean_text_classification_errors": "clean_reference_text_diagnostic",
                     "reconstructed_text_classification_errors": "reconstructed_asr_text_diagnostic",
                     "pipeline_sentence_errors_on_scored_only": "pipeline_sentence_strict"}.get(name)
        detail = ("chưa chấm" if score_key and not report["metrics"][score_key]["scored"]
                  else ", ".join(ids) if ids else "không có")
        lines.append(f"- {name}: {detail}.")
    for row in report["sentences"]:
        lines += ["", f"### {row['id']} — {row['true_label']}", "", f"Câu gốc: {cell(row['text'])}",
                  "", f"Lý do gán nhãn: {cell(row['reason'])}", "", f"Whisper sau đối chiếu: {cell(row['transcript']) or '(thiếu)'}",
                  "", f"Mốc: {row['start']}–{row['end']} giây; ước lượng: {row['timestamp_is_approximate']}. Đơn vị dùng chung: {row['has_shared_segment']}.",
                  "", row["diagnosis"]]
        for key, title in (("clean_text_prediction", "Hybrid trên câu gốc"), ("reconstructed_text_prediction", "Hybrid trên câu ASR ghép (chẩn đoán)")):
            prediction = row[key]
            if prediction:
                pho = prediction["phobert_result"]
                lines += ["", f"{title}: {prediction['final_label']}; PhoBERT: {pho['label']} ({pho['confidence']:.2%}); "
                          f"xác suất PhoBERT của nhãn cuối: {prediction['final_confidence']:.2%}.", "", prediction["explanation"]]
    lines += ["", "Confidence Hybrid vẫn là xác suất PhoBERT của nhãn cuối, chưa phải điểm đã hiệu chỉnh.",
              "Bộ 9 câu do trợ lý đề xuất, chưa được mô tả là nhãn đã có xác nhận độc lập của con người. "
              "Không dùng bộ này để train hoặc chỉnh model, Rules, ngưỡng. Không chạy lại final test v03.", ""]
    return "\n".join(lines)


def write_alignment_csv(path, rows):
    fields = ["id", "text", "true_label", "reason", "transcript", "start", "end", "segment_indexes", "unit_ids", "source_segment_ids", "timestamp_is_approximate",
              "alignment_status", "has_shared_segment", "word_similarity", "wer", "asr_text_status",
              "pipeline_segment_labels", "pipeline_all_segments_correct", "clean_text_label", "reconstructed_text_label", "diagnosis"]
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = {name: row[name] for name in fields if name in row}
            record.update({"word_similarity": row["text_metrics"]["word_similarity"], "wer": row["text_metrics"]["wer"],
                           "segment_indexes": json.dumps(row["segment_indexes"]),
                           "unit_ids": json.dumps(row["unit_ids"]), "source_segment_ids": json.dumps(row["source_segment_ids"]),
                           "pipeline_segment_labels": json.dumps(row["pipeline_segment_labels"]),
                           "clean_text_label": (row["clean_text_prediction"] or {}).get("final_label"),
                           "reconstructed_text_label": (row["reconstructed_text_prediction"] or {}).get("final_label")})
            writer.writerow(record)


def evaluate(source_path, ground_truth_path, device="auto", alignment_only=False):
    started = utc_now()
    source_hash, truth_hash = sha256(source_path), sha256(ground_truth_path)
    protected = {name: sha256(ROOT / name) for name in PROTECTED_CODE}
    rows = load_ground_truth(ground_truth_path)
    segments, saved_predictions, source = load_source(source_path)
    alignment = align_sentences(rows, segments)
    infer = None
    inference_calls = 0
    checkpoint = None
    if not alignment_only and alignment["global_alignment_eligible"]:
        checkpoint = verify_checkpoint()
        from nlp.hybrid_detector import HybridDetector, SAFE_THRESHOLD, WARNING_THRESHOLD
        from nlp.phobert_detector import PhoBertDetector
        if (WARNING_THRESHOLD, SAFE_THRESHOLD) != (0.55, 0.65):
            raise ValueError("Ngưỡng Hybrid đã thay đổi.")
        detector = HybridDetector(phobert_detector=PhoBertDetector(checkpoint_path=CHECKPOINT, device=None if device == "auto" else device))
        print(f"Checkpoint thực tế: {detector.phobert_detector.checkpoint_path}", flush=True)
        cache = {}

        def infer(text):
            nonlocal inference_calls
            if text not in cache:
                cache[text] = detector.predict(text)
                inference_calls += 1
            return cache[text]

        if saved_predictions is None:
            saved_predictions = [infer(s["text"]) if s["text"].strip() else None for s in segments]
    scored = score_alignment(alignment, segments, saved_predictions, infer, source.get("_evaluation_unit_kind", "whisper_segments_legacy"))
    scored["alignment_summary"]["whisper_segment_count"] = source.get("_whisper_segment_count", len(segments))
    if source_hash != sha256(source_path) or truth_hash != sha256(ground_truth_path):
        raise RuntimeError("Đầu vào đã thay đổi trong khi đánh giá.")
    if protected != {name: sha256(ROOT / name) for name in PROTECTED_CODE} or (checkpoint is not None and checkpoint != verify_checkpoint()):
        raise RuntimeError("Detector/checkpoint đã thay đổi trong khi đánh giá.")
    mode = "DIAGNOSTICS_AND_PIPELINE" if infer else "ALIGNMENT_ONLY" if alignment_only else "ALIGNMENT_REVIEW_REQUIRED_NO_MODEL_RUN"
    output = ROOT / "reports" / ("audio_pipeline_eval_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 2, "mode": mode, "started_at_utc": started, "finished_at_utc": utc_now(),
              "source_path": str(source_path), "source_sha256": source_hash,
              "ground_truth_path": str(ground_truth_path), "ground_truth_sha256": truth_hash,
              "checkpoint": checkpoint, "source_audio": source.get("audio_path"),
              "saved_segment_predictions_reused": "checkpoint" in source,
              "new_hybrid_inference_calls": inference_calls, "whisper_calls": 0,
              "alignment": {key: value for key, value in alignment.items() if key != "sentences"},
              "alignment_min_similarity": ALIGNMENT_MIN_SIMILARITY, "protected_code_sha256": protected,
              **scored}
    write_json(output / "report.json", report)
    write_alignment_csv(output / "alignment.csv", scored["sentences"])
    markdown = render_report(report)
    with (output / "report.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(markdown)
    print(markdown)
    print(f"Báo cáo: {output}", flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", help="result.json từ main.py (khuyến nghị) hoặc whisper.json.")
    parser.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--alignment-only", action="store_true", help="Không nạp model; xem ghép câu và kết quả segment đã lưu nếu có.")
    parser.add_argument("--validate-data", action="store_true", help="Chỉ kiểm tra 9 câu và không trùng v02/v03; không nạp model.")
    args = parser.parse_args(argv)
    if not args.validate_data and args.source is None:
        parser.error("Cần đường dẫn result.json/whisper.json, hoặc dùng --validate-data.")
    try:
        if args.validate_data:
            print(json.dumps(audit_ground_truth(args.ground_truth.resolve()), ensure_ascii=False, indent=2))
        else:
            evaluate(args.source.resolve(), args.ground_truth.resolve(), args.device, args.alignment_only)
        return 0
    except Exception as error:
        print(f"Lỗi kiểm thử audio: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
