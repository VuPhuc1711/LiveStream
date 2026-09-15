"""Kiểm tra ghép pipeline bằng dữ liệu giả; không đọc tập test hoặc nạp trọng số."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import main
from speech.sentence_units import build_classification_units


class AudioDemoTests(unittest.TestCase):
    def test_pipeline_loads_once_and_preserves_outputs(self):
        raw = {"text": " Câu giả lập. ", "language": "vi", "segments": [
            {"id": 0, "start": 0, "end": 1.2, "text": " Câu giả lập. "},
            {"id": 1, "start": 1.2, "end": 2.3, "text": "Câu giả thứ hai."},
            {"id": 2, "start": 2.3, "end": 3, "text": "   "},
        ]}
        prediction = {"final_label": "CAN_XAC_MINH", "final_confidence": 0.1,
                      "phobert_result": {"label": "KHONG_CANH_BAO", "confidence": 0.6,
                                         "probabilities": {"KHONG_CANH_BAO": 0.6, "CAN_XAC_MINH": 0.1, "CANH_BAO": 0.3}},
                      "rule_result": {"needs_review": False, "findings": []}, "explanation": "Giải thích giả."}
        with tempfile.TemporaryDirectory(prefix="audio_demo_test_") as temp:
            root = Path(temp)
            audio = root / "audio có dấu.mp3"
            audio.write_bytes(b"fake audio - never decoded")
            checkpoint = root / "models" / "fake" / "best"
            with (patch.object(main, "ROOT", root), patch.object(main, "CHECKPOINT", checkpoint),
                  patch.object(main, "check_environment", return_value={"cuda_available": False}),
                  patch.object(main, "verify_checkpoint", return_value={"path": str(checkpoint)}),
                  patch.object(main, "sha256", return_value="fake-hash"),
                  patch("speech.transcribe.WhisperTranscriber") as whisper_class,
                  patch("nlp.phobert_detector.PhoBertDetector") as phobert_class,
                  patch("nlp.hybrid_detector.HybridDetector") as hybrid_class,
                  contextlib.redirect_stdout(io.StringIO())):
                whisper_class.return_value.transcribe.return_value = raw
                phobert_class.return_value.checkpoint_path = checkpoint
                hybrid = hybrid_class.return_value
                hybrid.phobert_detector = phobert_class.return_value
                hybrid.predict.return_value = prediction
                result_path = main.run_audio(audio)
                whisper_class.assert_called_once_with(model_name="small", device="cpu")
                whisper_class.return_value.transcribe.assert_called_once_with(str(audio.resolve()))
                whisper_class.return_value.close.assert_called_once()
                phobert_class.assert_called_once_with(checkpoint_path=checkpoint, device="cpu")
                hybrid_class.assert_called_once_with(phobert_detector=phobert_class.return_value)
                self.assertEqual([call.args[0] for call in hybrid.predict.call_args_list],
                                 [raw["segments"][0]["text"].strip(), raw["segments"][1]["text"].strip()])
                report = json.loads(result_path.read_text(encoding="utf-8"))
                self.assertEqual(report["whisper_segments"], raw["segments"])
                self.assertEqual(report["segments"], report["classification_units"])
                self.assertEqual(len(report["classification_units"]), 2)
                self.assertEqual(report["segments"][0]["start"], 0)
                self.assertEqual(report["segments"][0]["end"], 1.2)
                self.assertEqual(report["segments"][0]["text"], raw["segments"][0]["text"].strip())
                self.assertEqual(report["segments"][0]["phobert_result"]["confidence"], 0.6)
                self.assertEqual(report["segments"][0]["final_confidence"], 0.1)
                self.assertEqual(report["summary"]["whisper_segments_without_units"], 1)
                self.assertEqual(report["classification_units"][0]["source_segment_id"], 0)
                self.assertEqual(json.loads((result_path.parent / "whisper.json").read_text(encoding="utf-8")), raw)

    def test_bad_audio_fails_before_environment_or_models(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(main, "check_environment") as check:
            with self.assertRaises(FileNotFoundError):
                main.run_audio(Path(temp) / "missing.mp3")
            check.assert_not_called()

    def test_two_sentences_are_two_predictions_with_approximate_timestamps(self):
        raw = {"segments": [{"id": 7, "start": 10.0, "end": 22.0, "text": "Mẫu này màu xanh. Hàng cũ nhưng nói mới!"}]}
        original = json.loads(json.dumps(raw))
        detector = MagicMock()
        detector.predict.side_effect = [{"final_label": "KHONG_CANH_BAO"}, {"final_label": "CANH_BAO"}]
        units = main.classify_units(raw, detector)
        self.assertEqual([call.args[0] for call in detector.predict.call_args_list],
                         ["Mẫu này màu xanh.", "Hàng cũ nhưng nói mới!"])
        self.assertEqual([u["final_label"] for u in units], ["KHONG_CANH_BAO", "CANH_BAO"])
        self.assertEqual([u["unit_id"] for u in units], ["U0001", "U0002"])
        self.assertTrue(all(u["source_segment_id"] == 7 and u["timestamp_is_approximate"] for u in units))
        self.assertEqual(units[0]["start"], 10)
        self.assertEqual(units[-1]["end"], 22)
        self.assertEqual(units[0]["end"], units[1]["start"])
        expected_end = 10 + 12 * len(units[0]["text"]) / sum(len(u["text"]) for u in units)
        self.assertAlmostEqual(units[0]["end"], expected_end)
        self.assertEqual(raw, original)

    def test_period_question_and_exclamation(self):
        text = 'Có màu đỏ. Còn hàng không? Mời xem nhé! “Đẹp quá!”'
        units = build_classification_units([{"id": 0, "start": 0, "end": 8, "text": text}])
        self.assertEqual([u["text"] for u in units], ['Có màu đỏ.', 'Còn hàng không?', 'Mời xem nhé!', '“Đẹp quá!”'])

    def test_empty_text_does_not_call_detector(self):
        detector = MagicMock()
        raw = {"segments": [{"id": i, "start": i, "end": i + 1, "text": text}
                            for i, text in enumerate(("", "   \n", "...?!"))]}
        self.assertEqual(main.classify_units(raw, detector), [])
        detector.predict.assert_not_called()

    def test_missing_terminal_punctuation_keeps_the_tail(self):
        for text, expected in (("Câu cuối chưa có dấu", ["Câu cuối chưa có dấu"]),
                               ("Câu đầu có dấu. Câu cuối chưa có dấu", ["Câu đầu có dấu.", "Câu cuối chưa có dấu"])):
            units = build_classification_units([{"id": 0, "start": 0, "end": 6, "text": text}])
            self.assertEqual([u["text"] for u in units], expected)
            self.assertEqual(units[-1]["end"], 6)
            self.assertEqual(units[0]["timestamp_is_approximate"], len(expected) > 1)

    def test_decimal_period_does_not_split_a_number(self):
        units = build_classification_units([{"id": 0, "start": 0, "end": 6, "text": "Dài 3.5 mét. Có sẵn nhé!"}])
        self.assertEqual([u["text"] for u in units], ["Dài 3.5 mét.", "Có sẵn nhé!"])

    def test_invalid_timestamps_are_rejected_before_prediction(self):
        detector = MagicMock()
        for start, end in ((2, 1), (-1, 2), (float("nan"), 3)):
            with self.assertRaises(ValueError):
                main.classify_segments({"segments": [{"start": start, "end": end, "text": "Giả lập"}]}, detector)
        detector.predict.assert_not_called()

    def test_whisper_is_called_once_with_explicit_precision(self):
        from speech.transcribe import WhisperTranscriber

        with tempfile.TemporaryDirectory() as temp:
            audio = Path(temp) / "fake.mp3"
            audio.write_bytes(b"fake")
            with patch("speech.transcribe.whisper.load_model") as loader, patch("speech.transcribe.shutil.which", return_value="ffmpeg"):
                model = loader.return_value
                transcriber = WhisperTranscriber(device="cpu")
                transcriber.transcribe(audio)
                loader.assert_called_once()
                model.transcribe.assert_called_once_with(str(audio.resolve()), language="vi", task="transcribe", fp16=False, verbose=False)
                transcriber.close()
                with self.assertRaises(RuntimeError):
                    transcriber.transcribe(audio)


if __name__ == "__main__":
    unittest.main()
