"""Streaming FFmpeg PCM and the same neural admission used by live microphone.

No normalization, subtitle blacklist or transcript-based admission. Windows retain
their original sample offsets; only contiguous forced-cut tails are joined.
"""
import subprocess
import tempfile
from contextlib import contextmanager

from backend.speech_buffer import RATE, UtteranceBuffer, Window

# Offline uploads need not cut speech every 12s for live latency. Leave room
# below Whisper's 30s context for frame rounding and a <1s contiguous tail.
WINDOW_SECONDS = 28
SHORT_TAIL_SECONDS = 1


class AudioDecodeError(ValueError):
    pass


@contextmanager
def decoded_pcm(path):
    # Stream instead of expanding a compressed upload into one large RAM array.
    # A temporary stderr sink prevents pipe deadlocks and never reaches the API.
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '0',
            '-i', str(path), '-f', 's16le', '-acodec', 'pcm_s16le', '-ac', '1',
            '-ar', str(RATE), 'pipe:1'], stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=errors)
        def chunks():
            found = False
            while pcm := process.stdout.read(RATE):  # 500ms; PCM16 sample boundaries
                if len(pcm) % 2:
                    raise AudioDecodeError('PCM giải mã không hợp lệ.')
                found = True
                yield pcm
            if process.wait() or not found:
                raise AudioDecodeError('File không chứa âm thanh hợp lệ hoặc đã bị hỏng.')
        try:
            yield chunks()
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
            process.wait()


class UploadSpeechWindows:
    def __init__(self, vad, max_seconds=None):
        self.buffer = UtteranceBuffer(max_seconds=WINDOW_SECONDS if max_seconds is None else max_seconds, vad=vad)
        self.received_samples = 0

    def windows(self, chunks):
        held = None
        def admit(window):
            nonlocal held
            previous = held
            # Offline look-ahead protects a sub-second continuation from being
            # decoded in isolation. Never concatenate across a skipped time gap.
            contiguous = previous is not None and (
                round(previous.start * RATE) + len(previous.pcm)//2 == round(window.start * RATE))
            if contiguous and not previous.boundary and len(window.pcm)//2 < RATE * SHORT_TAIL_SECONDS:
                held = Window(previous.pcm + window.pcm, previous.start,
                    window.speech_end, window.boundary, previous.voiced_samples + window.voiced_samples)
                return None
            held = window
            return previous
        for pcm in chunks:
            self.received_samples += len(pcm)//2
            for window in self.buffer.feed(pcm):
                previous = admit(window)
                if previous is not None:
                    yield previous
        for window in self.buffer.finish():
            previous = admit(window)
            if previous is not None:
                yield previous
        if held is not None:
            yield held
