"""Test phép ghép/chấm bằng fixture giả, không nạp model hoặc đọc final test."""

import contextlib
import copy
import csv
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import evaluate_audio_pipeline as evaluation
from speech.sentence_units import build_classification_units


def references():
    return [{"id": "R1", "text": "Chiếc ba lô màu xanh ở bàn.", "true_label": evaluation.LABELS[0], "reason": "Giả lập."},
            {"id": "R2", "text": "Mẫu đèn bàn tiết kiệm điện.", "true_label": evaluation.LABELS[1], "reason": "Giả lập."},
            {"id": "R3", "text": "Tôi biết hàng cũ nhưng nói mới.", "true_label": evaluation.LABELS[2], "reason": "Giả lập."}]


def segments(texts):
    return [{"id": i, "start": i * 4.0, "end": i * 4.0 + 3, "text": text} for i, text in enumerate(texts)]


def prediction(label):
    probabilities = {name: 0.8 if name == label else 0.1 for name in evaluation.LABELS}
    return {"final_label": label, "final_confidence": 0.8,
            "phobert_result": {"label": label, "confidence": 0.8, "probabilities": probabilities},
            "rule_result": {"needs_review": False, "findings": []}, "explanation": "Giải thích giả."}


class AudioEvaluationTests(unittest.TestCase):
    def test_new_units_override_legacy_segment_predictions_and_old_reports_still_load(self):
        raw = segments(["Mẫu màu xanh. Cần xem thông số?"])
        units = build_classification_units(raw)
        for index, unit in enumerate(units):
            unit.update(status="CLASSIFIED", **prediction(evaluation.LABELS[index]))
        legacy = [{**raw[0], "status": "CLASSIFIED", **prediction(evaluation.LABELS[2])}]
        source = {"status": "COMPLETED", "segments": legacy, "whisper_segments": raw,
                  "classification_units": units,
                  "checkpoint": {"actual_loaded_path": str(evaluation.CHECKPOINT), "file_sha256": {"fake": "hash"}},
                  "hybrid_thresholds": {"CANH_BAO": 0.55, "KHONG_CANH_BAO": 0.65},
                  "protected_code_sha256": {name: "fake" for name in evaluation.PROTECTED_CODE}}
        with tempfile.TemporaryDirectory() as temp, patch.object(evaluation, "verify_checkpoint", return_value={"file_sha256": {"fake": "hash"}}), patch.object(evaluation, "sha256", return_value="fake"):
            path = Path(temp) / "result.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            loaded, predictions, metadata = evaluation.load_source(path)
            self.assertEqual([u["text"] for u in loaded], ["Mẫu màu xanh.", "Cần xem thông số?"])
            self.assertEqual([p["final_label"] for p in predictions], list(evaluation.LABELS[:2]))
            self.assertEqual(metadata["_evaluation_unit_kind"], "classification_units")
            self.assertEqual(metadata["_whisper_segment_count"], 1)
            # Khi units rỗng, không dùng lại dự đoán nguyên segment làm fallback.
            source["classification_units"] = []
            path.write_text(json.dumps(source), encoding="utf-8")
            self.assertEqual(evaluation.load_source(path)[0], [])
            del source["classification_units"]
            del source["whisper_segments"]
            path.write_text(json.dumps(source), encoding="utf-8")
            loaded, predictions, metadata = evaluation.load_source(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(predictions[0]["final_label"], evaluation.LABELS[2])
            self.assertEqual(metadata["_evaluation_unit_kind"], "whisper_segments_legacy")

    def test_raw_whisper_input_also_splits_before_classification(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "whisper.json"
            path.write_text(json.dumps({"segments": segments(["Câu một. Câu hai chưa có dấu"])}), encoding="utf-8")
            loaded, predictions, metadata = evaluation.load_source(path)
            self.assertEqual([u["text"] for u in loaded], ["Câu một.", "Câu hai chưa có dấu"])
            self.assertIsNone(predictions)
            self.assertEqual(metadata["_evaluation_unit_kind"], "classification_units")

    def test_two_units_from_same_raw_segment_are_scored_independently(self):
        rows = references()[:2]
        units = build_classification_units(segments([rows[0]["text"] + " " + rows[1]["text"]]))
        aligned = evaluation.align_sentences(rows, units)
        result = evaluation.score_alignment(aligned, units, [prediction(row["true_label"]) for row in rows], unit_kind="classification_units")
        self.assertEqual(result["metrics"]["pipeline_classification_units"]["correct"], 2)
        self.assertEqual(result["alignment_summary"]["expected_sentence_count"], 2)
        self.assertEqual(result["alignment_summary"]["classification_unit_count"], 2)
        self.assertEqual(result["alignment_summary"]["unscorable_reference_ids"], [])
        self.assertEqual(result["alignment_summary"]["shared_unit_ids"], [])
        self.assertEqual(result["sentences"][0]["unit_ids"], ["U0001"])
        self.assertTrue(result["sentences"][0]["timestamp_is_approximate"])

    def test_multiple_units_must_not_claim_exact_timestamps(self):
        raw = segments(["Câu đầu. Câu sau!"])
        units = build_classification_units(raw)
        units[0]["timestamp_is_approximate"] = False
        with self.assertRaises(ValueError):
            evaluation.validate_classification_units(units, raw)

    def test_exact_alignment_and_original_segment_metrics(self):
        rows = references()
        source = segments([row["text"] for row in rows])
        aligned = evaluation.align_sentences(rows, source)
        result = evaluation.score_alignment(aligned, source, [prediction(row["true_label"]) for row in rows])
        self.assertEqual(aligned["segment_owners"], [[0], [1], [2]])
        self.assertEqual(aligned["global_text_metrics"]["wer"], 0)
        self.assertEqual(result["metrics"]["pipeline_original_segments"]["accuracy"], 1)
        self.assertEqual(result["metrics"]["pipeline_sentence_strict"]["correct"], 3)
        self.assertIsNone(result["metrics"]["clean_reference_text_diagnostic"]["accuracy"])

    def test_split_sentence_uses_all_original_segments_without_vote(self):
        rows = references()
        source = segments(["Chiếc ba lô", "màu xanh ở bàn.", rows[1]["text"], rows[2]["text"]])
        aligned = evaluation.align_sentences(rows, source)
        labels = [rows[0]["true_label"], rows[1]["true_label"], rows[1]["true_label"], rows[2]["true_label"]]
        scored = evaluation.score_alignment(aligned, source, list(map(prediction, labels)))
        self.assertEqual(aligned["sentences"][0]["segment_indexes"], [0, 1])
        self.assertTrue(aligned["sentences"][0]["text_metrics"]["normalized_exact_match"])
        self.assertFalse(scored["sentences"][0]["pipeline_all_segments_correct"])
        self.assertEqual(scored["metrics"]["pipeline_original_segments"]["accuracy"], 0.75)

    def test_merged_segment_does_not_duplicate_one_label_for_two_references(self):
        rows = references()
        source = segments([rows[0]["text"] + " " + rows[1]["text"], rows[2]["text"]])
        aligned = evaluation.align_sentences(rows, source)
        scored = evaluation.score_alignment(aligned, source, [prediction(rows[0]["true_label"]), prediction(rows[2]["true_label"])])
        self.assertEqual(aligned["segment_owners"], [[0, 1], [2]])
        self.assertTrue(aligned["sentences"][0]["text_metrics"]["normalized_exact_match"])
        self.assertTrue(aligned["sentences"][1]["text_metrics"]["normalized_exact_match"])
        self.assertIsNone(scored["sentences"][0]["pipeline_all_segments_correct"])
        self.assertEqual(scored["metrics"]["pipeline_original_segments"]["scored"], 1)
        self.assertEqual(scored["metrics"]["pipeline_sentence_strict"]["correct"], 1)

    def test_missing_middle_reference_does_not_shift_later_sentence(self):
        rows = references()
        aligned = evaluation.align_sentences(rows, segments([rows[0]["text"], rows[2]["text"]]))
        self.assertEqual(aligned["sentences"][1]["alignment_status"], "MISSING")
        self.assertEqual(aligned["sentences"][2]["segment_indexes"], [1])
        self.assertTrue(aligned["sentences"][2]["text_metrics"]["normalized_exact_match"])

    def test_unrelated_and_silent_audio_are_not_scored_as_good_matches(self):
        for source in ([], segments(["zzzz qqqq vvvv"]), segments(["   "])):
            aligned = evaluation.align_sentences(references(), source)
            self.assertFalse(aligned["global_alignment_eligible"])
            self.assertTrue(all(not r["alignment_eligible_for_scoring"] for r in aligned["sentences"]))

    def test_alignment_does_not_use_labels(self):
        rows = references()
        source = segments([row["text"] for row in rows])
        first = evaluation.align_sentences(rows, source)
        second_rows = copy.deepcopy(rows)
        for row in second_rows:
            row["true_label"] = "NOT_USED_BY_ALIGNER"
        second = evaluation.align_sentences(second_rows, source)
        self.assertEqual(first["segment_owners"], second["segment_owners"])
        self.assertEqual(first["global_text_metrics"], second["global_text_metrics"])

    def test_negation_is_preserved_and_wer_can_exceed_one(self):
        self.assertNotEqual(evaluation.normalized_text("không hết hạn"), evaluation.normalized_text("hết hạn"))
        self.assertEqual(evaluation.text_metrics("không hết hạn", "hết hạn")["deletions"], 1)
        self.assertGreater(evaluation.text_metrics("hàng", "hàng một hai ba")["wer"], 1)
        self.assertTrue(evaluation.text_metrics(" ĐÈN, bàn! ", "đèn bàn")["normalized_exact_match"])
        self.assertFalse(evaluation.text_metrics("tám giờ", "8 giờ")["normalized_exact_match"])

    def test_clean_classifier_error_is_separate_from_asr_error(self):
        rows = references()
        source = segments([row["text"] for row in rows])
        bad = prediction(evaluation.LABELS[2])
        aligned = evaluation.align_sentences(rows, source)
        scored = evaluation.score_alignment(aligned, source, [bad] * 3, lambda text: bad)
        self.assertEqual(scored["error_lists"]["whisper_text_differences_need_listening"], [])
        self.assertEqual(scored["error_lists"]["clean_text_classification_errors"], ["R1", "R2"])
        self.assertIn("ngay cả khi chưa có ASR", scored["sentences"][0]["diagnosis"])

    def test_asr_text_change_is_only_suspected_cause(self):
        rows = references()
        source = segments([rows[0]["text"].replace("xanh", "đỏ"), rows[1]["text"], rows[2]["text"]])
        def fake_predict(text):
            return prediction(evaluation.LABELS[2] if "đỏ" in text else next(row["true_label"] for row in rows if evaluation.normalized_text(row["text"]) == evaluation.normalized_text(text)))
        scored = evaluation.score_alignment(evaluation.align_sentences(rows, source), source,
                                             [fake_predict(s["text"]) for s in source], fake_predict)
        self.assertEqual(scored["error_lists"]["clean_text_classification_errors"], [])
        self.assertEqual(scored["error_lists"]["reconstructed_text_classification_errors"], ["R1"])
        self.assertIn("nghi ảnh hưởng", scored["sentences"][0]["diagnosis"])

    def test_prediction_wrong_confidence_and_policy_are_rejected(self):
        valid = prediction(evaluation.LABELS[0])
        self.assertEqual(evaluation.validate_prediction(valid), valid)
        for key, value in (("final_confidence", 0.1), ("final_label", evaluation.LABELS[1])):
            invalid = copy.deepcopy(valid)
            invalid[key] = value
            with self.assertRaises(ValueError):
                evaluation.validate_prediction(invalid)

    def test_full_alignment_only_export_without_model_loading(self):
        with tempfile.TemporaryDirectory(prefix="audio_eval_fake_") as temp:
            root = Path(temp)
            csv_path, json_path = root / "truth.csv", root / "whisper.json"
            rows = [{**row, "id": f"FAKE_{group}_{index}", "text": f"Mẫu giả {group} " + row["text"]}
                    for group in range(3) for index, row in enumerate(references())]
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["id", "text", "true_label", "reason"])
                writer.writeheader()
                writer.writerows(rows)
            json_path.write_text(json.dumps({"segments": segments([r["text"] for r in rows])}, ensure_ascii=False), encoding="utf-8")
            with patch.object(evaluation, "ROOT", root), patch.object(evaluation, "PROTECTED_CODE", ()), contextlib.redirect_stdout(io.StringIO()):
                folder = evaluation.evaluate(json_path, csv_path, alignment_only=True)
            report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["mode"], "ALIGNMENT_ONLY")
            self.assertEqual(report["new_hybrid_inference_calls"], 0)
            self.assertEqual(report["whisper_calls"], 0)
            self.assertEqual(len(report["sentences"]), 9)
            with (folder / "alignment.csv").open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 9)
            self.assertIn("chưa chấm", (folder / "report.md").read_text(encoding="utf-8"))

    def test_full_diagnostics_loads_once_and_reuses_saved_segment_decisions(self):
        for use_saved in (False, True):
            with self.subTest(use_saved=use_saved), tempfile.TemporaryDirectory(prefix="audio_eval_mock_") as temp:
                root = Path(temp)
                truth_path, source_path = root / "truth.csv", root / "source.json"
                rows = [{**row, "id": f"FAKE_{group}_{index}", "text": f"Mẫu giả {group} " + row["text"]}
                        for group in range(3) for index, row in enumerate(references())]
                with truth_path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["id", "text", "true_label", "reason"])
                    writer.writeheader()
                    writer.writerows(rows)
                source_path.write_text("{}", encoding="utf-8")
                source_segments = segments([r["text"] for r in rows])
                saved = [prediction(evaluation.LABELS[1]) for _ in rows] if use_saved else None
                fake_phobert, fake_hybrid = types.ModuleType("nlp.phobert_detector"), types.ModuleType("nlp.hybrid_detector")
                fake_phobert.PhoBertDetector = MagicMock()
                fake_phobert.PhoBertDetector.return_value.checkpoint_path = evaluation.CHECKPOINT
                hybrid = MagicMock()
                hybrid.phobert_detector = fake_phobert.PhoBertDetector.return_value
                lookup = {row["text"]: row["true_label"] for row in rows}
                hybrid.predict.side_effect = lambda text: prediction(lookup[text])
                fake_hybrid.HybridDetector = MagicMock(return_value=hybrid)
                fake_hybrid.SAFE_THRESHOLD, fake_hybrid.WARNING_THRESHOLD = 0.65, 0.55
                with (patch.object(evaluation, "ROOT", root), patch.object(evaluation, "PROTECTED_CODE", ()),
                      patch.object(evaluation, "verify_checkpoint", return_value={"path": "fake"}),
                      patch.object(evaluation, "load_source", return_value=(source_segments, saved, {"checkpoint": {}} if use_saved else {})),
                      patch.dict("sys.modules", {"nlp.phobert_detector": fake_phobert, "nlp.hybrid_detector": fake_hybrid}),
                      contextlib.redirect_stdout(io.StringIO())):
                    folder = evaluation.evaluate(source_path, truth_path)
                fake_phobert.PhoBertDetector.assert_called_once_with(checkpoint_path=evaluation.CHECKPOINT, device=None)
                fake_hybrid.HybridDetector.assert_called_once_with(phobert_detector=fake_phobert.PhoBertDetector.return_value)
                report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
                self.assertEqual(report["new_hybrid_inference_calls"], 9)
                self.assertEqual(report["whisper_calls"], 0)
                self.assertEqual(report["metrics"]["clean_reference_text_diagnostic"]["accuracy"], 1)
                self.assertEqual(report["metrics"]["pipeline_original_segments"]["accuracy"], 1 / 3 if use_saved else 1)
                self.assertEqual(report["saved_segment_predictions_reused"], use_saved)


if __name__ == "__main__":
    unittest.main()
