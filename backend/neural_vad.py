"""CPU speech detector loaded once. Input is unchanged 16 kHz mono PCM16.

Load bundled TorchScript directly: no torchaudio, download, normalization or
change to the existing PyTorch installation. The resource gate permits one live
session; reset recurrent state at its start.
"""
import hashlib
from importlib.metadata import distribution
from pathlib import Path

import numpy as np
import torch


class NeuralVad:
    frame_samples = 512
    onset_threshold = .5
    offset_threshold = .35
    min_speech_ms = 250

    def __init__(self):
        package = distribution('silero-vad')
        if package.version != '6.2.1':
            raise RuntimeError('Install requirements-vad.txt with --no-deps (silero-vad==6.2.1).')
        path = Path(package.locate_file('silero_vad/data/silero_vad.jit'))
        self.model_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        self.model = torch.jit.load(str(path), map_location='cpu').eval()
        self.reset()

    def reset(self):
        self.model.reset_states()

    def probability(self, pcm):
        values = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768
        if len(values) > self.frame_samples:
            raise ValueError('VAD frame exceeds 512 samples')
        values = np.pad(values, (0, self.frame_samples-len(values)))
        with torch.inference_mode():
            return float(self.model(torch.from_numpy(values), 16000))

    def configuration(self):
        return dict(implementation='silero_vad_6.2.1', revision='neural_admission_v1',
            device='cpu', model_sha256=self.model_hash, frame_samples=self.frame_samples,
            sample_rate=16000, onset_threshold=self.onset_threshold,
            offset_threshold=self.offset_threshold, min_speech_ms=self.min_speech_ms,
            candidate_gap_ms=128, onset_lookback_ms=300,
            pre_post_roll_seconds=.3, boundary_silence_seconds=1.5)
