"""Single source of the active pipeline checkpoint; no ML imports."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / 'pipeline_model.json'


def get_active_model():
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    if not config.get('version') or not config.get('checkpoint'):
        raise ValueError('Cấu hình model thiếu version hoặc checkpoint.')
    return config


def get_default_checkpoint():
    return (ROOT / get_active_model()['checkpoint']).resolve()


def verify_model_checkpoint(checkpoint_path=None):
    checkpoint = get_default_checkpoint() if checkpoint_path is None else Path(checkpoint_path).resolve()
    lock_path = checkpoint.parent / 'checkpoint_lock.json'
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    if lock.get('status') != 'FROZEN' or Path(lock['checkpoint']).resolve() != checkpoint:
        raise ValueError('Bản khóa không trỏ tới checkpoint được yêu cầu.')
    if 'file_sha256' in lock:
        expected = lock['file_sha256']
    else:
        expected = {Path(name).relative_to('best').as_posix(): digest
                    for name, digest in lock['artifact_sha256'].items() if Path(name).parts[0] == 'best'}
    actual = {}
    for path in sorted(checkpoint.rglob('*')):
        if path.is_file():
            with path.open('rb') as stream:
                actual[path.relative_to(checkpoint).as_posix()] = hashlib.file_digest(stream, 'sha256').hexdigest()
    if not expected or actual != expected:
        raise ValueError('Checkpoint không khớp SHA-256 đã khóa; dừng nạp model.')
    return {'path': str(checkpoint), 'file_sha256': actual, 'verified_against': str(lock_path)}
