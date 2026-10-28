"""Shared model startup helpers; independent of historical research/demo scripts."""

import importlib
import importlib.metadata
import shutil
import subprocess
import sys
from pathlib import Path

from backend.model_config import get_default_checkpoint, verify_model_checkpoint
from speech.sentence_units import build_classification_units


CHECKPOINT = get_default_checkpoint()


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
    """Một lần dự đoán PhoBERT cho mỗi câu; raw Whisper segments không bị sửa."""
    units = build_classification_units(whisper_result["segments"])
    for unit in units:
        unit.update({"status": "CLASSIFIED", **detector.predict(unit["text"])})
    return units


def classify_segments(whisper_result, detector):
    """Tên hàm tương thích cũ; kết quả nay là các unit câu, không phải raw segment."""
    return classify_units(whisper_result, detector)


def create_detector(checkpoint_path=None, device=None, classification_mode='three_class'):
    """Đường nạp dùng chung cho demo và smoke test, một PhoBERT cho mọi unit."""
    from nlp.phobert_detector import PhoBertDetector
    from nlp.model_only_detector import ModelOnlyDetector
    selected = CHECKPOINT if checkpoint_path is None else Path(checkpoint_path).resolve()
    options = {'classification_mode': classification_mode} if classification_mode != 'three_class' else {}
    detector = ModelOnlyDetector(phobert_detector=PhoBertDetector(checkpoint_path=selected, device=device, **options))
    if detector.phobert_detector.checkpoint_path.resolve() != selected.resolve():
        raise RuntimeError("PhoBERT nạp sai checkpoint được yêu cầu.")
    return detector
