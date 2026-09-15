"""Own models for one application lifespan; reuse existing inference logic."""
import logging
import threading
from collections import Counter

from main import check_environment, classify_units, create_detector, verify_checkpoint
from model_config import get_active_model, get_default_checkpoint


class InvalidAudioError(ValueError):
    pass


class PipelineService:
    def __init__(self):
        self.checkpoint = get_active_model()['checkpoint']
        self.transcriber = None
        self.detector = None
        self.lock = threading.Lock()

    def start(self):
        from speech.transcribe import WhisperTranscriber
        environment = check_environment()
        checkpoint = get_default_checkpoint()
        verify_checkpoint(checkpoint)
        logging.getLogger(__name__).info('Verified checkpoint: %s', checkpoint)
        # Keep Whisper on CPU so both persistent models fit on a 4 GiB GPU.
        self.transcriber = WhisperTranscriber('small', device='cpu')
        self.detector = create_detector(checkpoint, 'cuda' if environment['cuda_available'] else 'cpu')
        logging.getLogger(__name__).info('Loaded checkpoint: %s', self.detector.phobert_detector.checkpoint_path)

    def health(self):
        ready = self.transcriber is not None and self.detector is not None
        return {'status': 'ok' if ready else 'not_ready', 'checkpoint': self.checkpoint,
                'whisper': 'ready' if self.transcriber is not None else 'not_ready',
                'phobert': 'ready' if self.detector is not None else 'not_ready', 'ready': ready}

    def analyze(self, path):
        with self.lock:
            if not self.health()['ready']:
                raise RuntimeError('Pipeline is not ready')
            try:
                raw = self.transcriber.transcribe(str(path))
            except RuntimeError as error:
                if str(error).startswith('Failed to load audio:'):
                    raise InvalidAudioError('Không đọc được nội dung âm thanh.') from error
                raise
            units = classify_units(raw, self.detector)
            counts = Counter(unit['final_label'] for unit in units)
            return {'transcript': raw['text'], 'classification_units': units,
                    'summary': {'whisper_segment_count': len(raw['segments']),
                                'classification_unit_count': len(units),
                                'final_label_counts': {label: counts[label] for label in
                                    ('KHONG_CANH_BAO', 'CAN_XAC_MINH', 'CANH_BAO')}}}

    def close(self):
        self.detector = None
        if self.transcriber is not None:
            self.transcriber.close()
            self.transcriber = None
