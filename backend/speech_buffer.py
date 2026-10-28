"""Session-local neural VAD, preserved PCM and lossless audio accounting."""
from collections import deque
from dataclasses import dataclass

RATE = 16000


@dataclass
class Window:
    pcm: bytes
    start: float
    speech_end: float
    boundary: bool
    voiced_samples: int = 0
    chunk_start: int | None = None
    chunk_end: int | None = None


class UtteranceBuffer:
    """Confirm acoustic speech before enqueueing ASR, including on finish().

    250 ms of neural speech evidence (.5 onset/.35 continuation) rejects brief
    noise impulses. 300 ms pre/post-roll preserves word attacks and endings.
    Silent gaps up to 128 ms may bridge evidence but do not count as voiced.
    """
    def __init__(self, max_seconds=12, vad=None, trace=None):
        if vad is None:
            from backend.neural_vad import NeuralVad
            vad = NeuralVad()
        self.vad = vad
        self.vad.reset()
        self.frame_samples = int(vad.frame_samples)
        self.frame_bytes = self.frame_samples * 2
        self.onset = float(vad.onset_threshold)
        self.offset = float(vad.offset_threshold)
        self.min_speech = round(vad.min_speech_ms * RATE / 1000)
        self.trace = trace
        self.max_samples = round(max_seconds * RATE)
        self.position = 0
        self.silence_samples = 0
        self.pre = deque()
        self.pre_samples = 0
        self.buffer = bytearray()
        self.remainder = bytearray()
        self.start = 0
        self.silent = 0
        self.speech_end = 0
        self.active = False
        self.continuation = False
        self.voiced_samples = 0
        self.candidate_start = None
        self.candidate_voiced = self.candidate_gap = 0
        self.recent_evidence = deque()

    def _begin_candidate(self, begin, length):
        self.candidate_start = begin
        self.candidate_voiced = length
        # A weak attack can precede a confident vowel. Confirm retrospectively
        # within pre-roll, without counting silence or lowering onset threshold.
        gap = 0
        for start, samples, probability in reversed(list(self.recent_evidence)[:-1]):
            if probability >= self.offset:
                self.candidate_start = start
                self.candidate_voiced += samples
                gap = 0
            else:
                gap += samples
                if gap >= RATE*.128:
                    break

    def _reset_candidate(self):
        self.candidate_start = None
        self.candidate_voiced = self.candidate_gap = 0

    def _trim_pre(self, keep):
        # Remove exact excess samples, not whole frames, to preserve time.
        excess = max(0, self.pre_samples-keep)
        self.silence_samples += excess
        while excess:
            first = self.pre.popleft()
            count = min(excess, len(first)//2)
            if count*2 < len(first):
                self.pre.appendleft(first[count*2:])
            self.pre_samples -= count
            excess -= count

    def _frame(self, frame):
        length = len(frame) // 2
        probability = self.vad.probability(frame)
        begin = self.position
        self.position += length
        self.recent_evidence.append((begin, length, probability))
        while self.recent_evidence and begin-self.recent_evidence[0][0] >= 4800:
            self.recent_evidence.popleft()
        voiced = probability >= self.offset if self.active or self.candidate_start is not None else probability >= self.onset
        admitted_now = rejected_candidate = False
        if not self.active:
            self.pre.append(frame)
            self.pre_samples += length
            if voiced:
                if self.candidate_start is None:
                    self._begin_candidate(begin, length)
                else:
                    self.candidate_voiced += length
                self.candidate_gap = 0
            elif self.candidate_start is not None:
                self.candidate_gap += length
                if self.candidate_gap >= RATE*.128:
                    rejected_candidate = True
                    self._reset_candidate()
            if self.candidate_start is not None and self.candidate_voiced >= self.min_speech:
                self._trim_pre(self.position-max(0, self.candidate_start-4800))
                self.start = self.position-self.pre_samples
                self.buffer.extend(b''.join(self.pre))
                self.pre.clear()
                self.pre_samples = 0
                self.active = admitted_now = True
                self.voiced_samples = self.candidate_voiced
                self.speech_end = self.position
                self._reset_candidate()
            else:
                keep = self.position-self.candidate_start+4800 if self.candidate_start is not None else 4800
                self._trim_pre(keep)
        else:
            self.buffer.extend(frame)
            if voiced:
                self.voiced_samples += length
                self.speech_end = self.position
        if self.trace:
            self.trace(dict(start=begin/RATE, end=self.position/RATE,
                speech_probability=probability, frame_is_speech=probability >= self.offset,
                speech_onset_evidence=probability >= self.onset,
                admitted=self.active, admitted_now=admitted_now,
                candidate_voiced_seconds=self.candidate_voiced/RATE,
                rejected_short_candidate=rejected_candidate))
        if not self.active:
            return None
        self.silent = 0 if voiced else self.silent + length
        if self.silent >= RATE * 1.5:
            result = self._take(True)
            self.active = False
            self.continuation = False
            self.silent = 0
            self._reset_candidate()
            return result
        if len(self.buffer)//2 >= self.max_samples and self.voiced_samples:
            result = self._take(False)
            self.continuation = True
            return result
        return None

    def feed(self, pcm):
        self.remainder.extend(pcm)
        windows = []
        while len(self.remainder) >= self.frame_bytes:
            frame = bytes(self.remainder[:self.frame_bytes])
            del self.remainder[:self.frame_bytes]
            window = self._frame(frame)
            if window is not None:
                windows.append(window)
        return windows

    def _take(self, boundary):
        pcm = bytes(self.buffer)
        if boundary:
            # Wait 1.5s to decide the boundary, but do not feed all that silence to ASR.
            keep = min(len(pcm)//2, max(0, self.speech_end + 4800 - self.start))
            if not self.voiced_samples:
                keep = 0  # Pure tail after a forced cut: flush text without another ASR.
            self.silence_samples += len(pcm)//2 - keep
            pcm = pcm[:keep*2]
        result = Window(pcm, self.start/RATE, self.speech_end/RATE, boundary, self.voiced_samples)
        self.buffer.clear()
        self.start = self.position
        self.voiced_samples = 0
        return result

    def finish(self):
        windows = []
        if self.remainder:
            window = self._frame(bytes(self.remainder))
            self.remainder.clear()
            if window is not None:
                windows.append(window)
        # End-of-session never bypasses admission for a short noise candidate.
        if self.buffer:
            windows.append(self._take(True))
        elif self.continuation:
            windows.append(Window(b'', self.position/RATE, self.speech_end/RATE, True))
        self.silence_samples += self.pre_samples
        self.pre.clear()
        self.pre_samples = 0
        self._reset_candidate()
        self.recent_evidence.clear()
        self.active = False
        self.continuation = False
        return windows
