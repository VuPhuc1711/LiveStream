"""Demo audio: Whisper một lần → tách câu → Hybrid với checkpoint cấu hình chung."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from speech.sentence_units import SEGMENTATION_VERSION, build_classification_units
from model_config import get_default_checkpoint, verify_model_checkpoint


ROOT = Path(__file__).resolve().parent
CHECKPOINT = get_default_checkpoint()
PROTECTED_CODE = ("nlp/keyword_detector.py", "nlp/phobert_detector.py", "nlp/hybrid_detector.py")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def check_environment():
    """Kiểm tra import thực tế và FFmpeg; không tự cài/nâng cấp dependency."""
    versions, modules, errors = {}, {}, []
    for name, distribution in (("torch", "torch"), ("whisper", "openai-whisper"),
                               ("transformers", "transformers"), ("underthesea", "underthesea")):
        try:
            modules[name] = importlib.import_module(name)
            versions[distribution] = importlib.metadata.version(distribution)
        except Exception as error:
            errors.append(f"{distribution}: {type(error).__name__}: {error}")
    ffmpeg = shutil.which("ffmpeg")
    ffmpeg_version = None
    if ffmpeg is None:
        errors.append("FFmpeg: không tìm thấy ffmpeg trong PATH (kiểm tra: ffmpeg -version).")
    else:
        try:
            result = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", check=True, timeout=15)
            ffmpeg_version = result.stdout.splitlines()[0]
        except (OSError, subprocess.SubprocessError, IndexError) as error:
            errors.append(f"FFmpeg tại {ffmpeg}: {error}")
    if errors:
        raise RuntimeError("Môi trường chưa chạy được demo:\n- " + "\n- ".join(errors)
                           + "\nDùng .venv-nlp; xem README. Không tự thay PyTorch CUDA đang hoạt động.")
    torch = modules["torch"]
    cuda_available = torch.cuda.is_available()
    try:
        tensor = torch.ones(2, device="cuda" if cuda_available else "cpu")
        if (tensor + 1).cpu().tolist() != [2.0, 2.0]:
            raise RuntimeError("Kết quả tensor không hợp lệ.")
        del tensor
        modules["underthesea"].word_tokenize("Đây là câu kiểm tra môi trường.", format="text")
    except Exception as error:
        raise RuntimeError(f"Kiểm tra thực thi PyTorch/Underthesea thất bại: {error}") from error
    return {
        "python_executable": sys.executable, "python_version": sys.version,
        "versions": versions, "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "ffmpeg_path": ffmpeg, "ffmpeg_version": ffmpeg_version,
    }


def verify_checkpoint(checkpoint_path=None):
    """Kiểm tra checkpoint đang chọn; hỗ trợ manifest v02 và v03."""
    return verify_model_checkpoint(CHECKPOINT if checkpoint_path is None else checkpoint_path)


def classify_units(whisper_result, detector):
    """Một lần Hybrid.predict cho mỗi câu; raw Whisper segments không bị sửa."""
    units = build_classification_units(whisper_result["segments"])
    for unit in units:
        unit.update({"status": "CLASSIFIED", **detector.predict(unit["text"])})
    return units


def classify_segments(whisper_result, detector):
    """Tên hàm tương thích cũ; kết quả nay là các unit câu, không phải raw segment."""
    return classify_units(whisper_result, detector)


def create_detector(checkpoint_path=None, device=None):
    """Đường nạp dùng chung cho demo và smoke test, một PhoBERT cho mọi unit."""
    from nlp.phobert_detector import PhoBertDetector
    from nlp.hybrid_detector import HybridDetector
    selected = CHECKPOINT if checkpoint_path is None else Path(checkpoint_path).resolve()
    detector = HybridDetector(phobert_detector=PhoBertDetector(checkpoint_path=selected, device=device))
    if detector.phobert_detector.checkpoint_path.resolve() != selected.resolve():
        raise RuntimeError("Hybrid nạp sai checkpoint được yêu cầu.")
    return detector


def run_audio(audio_path, device="auto", checkpoint_path=None):
    audio_path = Path(audio_path).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file âm thanh: {audio_path}")
    started_at, started_clock = utc_now(), time.perf_counter()
    environment = check_environment()
    if device == "cuda" and not environment["cuda_available"]:
        raise RuntimeError("CUDA không khả dụng. Chọn --device cpu để chạy trên CPU.")
    actual_device = ("cuda" if environment["cuda_available"] else "cpu") if device == "auto" else device
    selected_checkpoint = CHECKPOINT if checkpoint_path is None else Path(checkpoint_path).expanduser().resolve()
    checkpoint_info = verify_checkpoint(selected_checkpoint)
    protected_before = {name: sha256(ROOT / name) for name in PROTECTED_CODE}
    audio_hash = sha256(audio_path)
    print("Môi trường:", json.dumps(environment, ensure_ascii=False), flush=True)
    print(f"Checkpoint PhoBERT (đã kiểm tra hash): {selected_checkpoint.resolve()}", flush=True)
    print(f"Audio: {audio_path}\nThời gian bắt đầu UTC: {started_at}", flush=True)

    from speech.transcribe import WhisperTranscriber
    from nlp.phobert_detector import PhoBertDetector
    from nlp.hybrid_detector import HybridDetector, SAFE_THRESHOLD, WARNING_THRESHOLD

    if (WARNING_THRESHOLD, SAFE_THRESHOLD) != (0.55, 0.65):
        raise ValueError("Ngưỡng Hybrid không còn đúng cấu hình đã chốt.")
    output = ROOT / "reports" / ("audio_demo_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    try:
        print(f"[1/2] Whisper small đang xử lý toàn bộ file trên {actual_device}...", flush=True)
        transcriber = WhisperTranscriber(model_name="small", device=actual_device)
        try:
            whisper_result = transcriber.transcribe(str(audio_path))  # Chỉ một lần cho toàn bộ file.
        finally:
            transcriber.close()
        write_json(output / "whisper.json", whisper_result)

        print("[2/2] Nạp PhoBERT một lần, tách câu và kiểm tra từng classification unit...", flush=True)
        detector = create_detector(selected_checkpoint, actual_device)
        actual_checkpoint = detector.phobert_detector.checkpoint_path.resolve()
        if actual_checkpoint != selected_checkpoint.resolve():
            raise RuntimeError(f"Hybrid nạp sai checkpoint: {actual_checkpoint}")
        print(f"Checkpoint thực tế Hybrid đã nạp: {actual_checkpoint}", flush=True)
        units = classify_units(whisper_result, detector)
        raw_segments = whisper_result["segments"]
        print(f"Whisper segments: {len(raw_segments)}; classification units: {len(units)}", flush=True)
        unchanged = (verify_checkpoint(selected_checkpoint) == checkpoint_info
                     and protected_before == {name: sha256(ROOT / name) for name in PROTECTED_CODE}
                     and sha256(audio_path) == audio_hash)
        if not unchanged:
            raise RuntimeError("Audio, checkpoint hoặc mã detector đã thay đổi trong khi chạy.")
        counts = Counter(row["final_label"] for row in units)
        report = {
            "schema_version": 2, "status": "COMPLETED" if counts else "NO_SPEECH", "started_at_utc": started_at,
            "finished_at_utc": utc_now(), "duration_seconds": time.perf_counter() - started_clock,
            "audio_path": str(audio_path), "audio_sha256": audio_hash, "environment": environment,
            "device": actual_device, "checkpoint": {**checkpoint_info, "actual_loaded_path": str(actual_checkpoint)},
            "whisper": {"model": "small", "language": whisper_result.get("language", "vi"),
                        "text": whisper_result["text"], "raw_result": str(output / "whisper.json")},
            "hybrid_thresholds": {"CANH_BAO": WARNING_THRESHOLD, "KHONG_CANH_BAO": SAFE_THRESHOLD},
            "confidence_notes": {
                "phobert_result.confidence": "Xác suất softmax của nhãn PhoBERT dự đoán; chưa hiệu chỉnh.",
                "final_label": "Quyết định cuối theo chính sách Hybrid; có thể khác nhãn PhoBERT.",
                "final_confidence": "Xác suất PhoBERT của final_label; không phải confidence tổng hợp hoặc độ tin cậy Whisper.",
                "rule_result": "Các cụm từ Rules khớp; Rules không cung cấp confidence.",
                "timestamps": "start/end tính bằng giây. Khi chia segment thành nhiều câu, phân bổ theo độ dài ký tự và timestamp_is_approximate=true. False chỉ có nghĩa giữ mốc segment Whisper.",
            },
            "segmentation": {"version": SEGMENTATION_VERSION, "boundary_source": "Whisper punctuation only; no ground truth",
                             "segments_compatibility_alias": "classification_units"},
            "whisper_segments": raw_segments, "classification_units": units,
            # Các công cụ cũ vẫn nhận danh sách có id/start/end/text và kết quả; raw ở whisper_segments.
            "segments": units, "summary": {"whisper_segment_count": len(raw_segments), "classification_unit_count": len(units),
                "total_segments": len(units), "classified_segments": len(units), "skipped_empty_segments": 0,
                "whisper_segments_without_units": len(raw_segments) - len({u["source_segment_index"] for u in units}),
                "final_label_counts": dict(counts)},
            "execution_counts": {"whisper_model_loads": 1, "whisper_transcribe_calls": 1,
                                 "phobert_model_loads": 1, "hybrid_predict_calls": sum(counts.values())},
            "protected_code_sha256": protected_before, "inputs_and_detectors_unchanged": unchanged,
        }
        write_json(output / "result.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
        print(f"\nĐã lưu JSON: {output / 'result.json'}", flush=True)
        return output / "result.json"
    except Exception as error:
        write_json(output / "error.json", {"status": "FAILED", "started_at_utc": started_at,
                   "failed_at_utc": utc_now(), "audio_path": str(audio_path), "checkpoint": checkpoint_info,
                   "error": f"{type(error).__name__}: {error}"})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", nargs="?", type=Path, help="Đường dẫn file audio (đặt trong dấu nháy nếu có khoảng trắng).")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Mặc định dùng CUDA nếu có.")
    parser.add_argument("--check-environment", action="store_true", help="Chỉ kiểm tra thư viện/CUDA/FFmpeg, không nạp model.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Ghi đè checkpoint cấu hình chung cho lần chạy này.")
    args = parser.parse_args(argv)
    if not args.check_environment and args.audio_path is None:
        parser.error("Cần đường dẫn audio, ví dụ: main.py .\\audio\\voicetiengviet.mp3")
    try:
        if args.check_environment:
            print(json.dumps(check_environment(), ensure_ascii=False, indent=2))
        else:
            run_audio(args.audio_path, args.device, args.checkpoint)
        return 0
    except Exception as error:
        print(f"Lỗi demo: {type(error).__name__}: {error}", file=sys.stderr)
        if "out of memory" in str(error).lower():
            print("GPU thiếu VRAM. Đóng ứng dụng dùng GPU hoặc chạy lại cùng lệnh với --device cpu.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
