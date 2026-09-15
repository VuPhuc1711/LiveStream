import gc
import shutil
from pathlib import Path

import torch
import whisper


class WhisperTranscriber:
    def __init__(self, model_name="small", device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Loading Whisper model: {model_name} ({self.device})", flush=True)
        self.model = whisper.load_model(model_name, device=self.device)
        self.model.eval()

    def transcribe(self, audio_path):
        if self.model is None:
            raise RuntimeError("Whisper đã được giải phóng; không thể transcribe tiếp.")
        audio_path = Path(audio_path).expanduser().resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file âm thanh: {audio_path}")
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("Không tìm thấy FFmpeg trong PATH. Kiểm tra bằng: ffmpeg -version")
        # Một lần gọi cho toàn bộ file; giữ text, start/end và các segment gốc của Whisper.
        with torch.inference_mode():
            return self.model.transcribe(
                str(audio_path),
                language="vi",
                task="transcribe",
                fp16=self.device.type == "cuda",
                verbose=False,
            )

    def close(self):
        """Giải phóng Whisper trước khi nạp PhoBERT trên GPU ít VRAM."""
        self.model = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    transcriber = WhisperTranscriber("small")

    result = transcriber.transcribe("audio/voicetiengviet.mp3")

    print("\n===== TRANSCRIPT =====")
    print(result["text"])
