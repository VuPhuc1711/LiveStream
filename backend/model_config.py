"""Single source of the active pipeline checkpoint; no ML imports."""
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'pipeline_model.json'


def get_active_model():
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    if not config.get('version') or not config.get('checkpoint'):
        raise ValueError('Cấu hình model thiếu version hoặc checkpoint.')
    return config


def get_default_checkpoint():
    return (ROOT / get_active_model()['checkpoint']).resolve()


def _locked_model_identity(value):
    """Read the original model-relative identity without resolving its old OS root.

    Frozen records intentionally retain their original absolute Windows paths.
    A clone/container may relocate the repository, but not the run/seed/checkpoint
    identity below models/. The immutable artifact hashes still verify every byte.
    """
    if not isinstance(value, str) or not value or '\x00' in value:
        raise ValueError('Đường dẫn trong bản khóa checkpoint không hợp lệ.')
    parts = PurePosixPath(value.replace('\\', '/')).parts
    if '..' in parts or parts.count('models') != 1:
        raise ValueError('Bản khóa thiếu định danh models/ hợp lệ.')
    identity = parts[parts.index('models'):]
    if len(identity) < 3:
        raise ValueError('Bản khóa thiếu định danh lượt train/checkpoint.')
    return identity


def _relative_artifact(value):
    if not isinstance(value, str) or not value or '\x00' in value:
        raise ValueError('Tên artifact trong bản khóa không hợp lệ.')
    path = PurePosixPath(value.replace('\\', '/'))
    if not path.parts or path.is_absolute() or '..' in path.parts or ':' in value:
        raise ValueError('Artifact phải có đường dẫn tương đối an toàn.')
    return path


def verify_model_checkpoint(checkpoint_path=None):
    checkpoint = get_default_checkpoint() if checkpoint_path is None else Path(checkpoint_path).resolve()
    lock_path = checkpoint.parent / 'checkpoint_lock.json'
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    try:
        identity = checkpoint.relative_to(ROOT.resolve()).parts
    except ValueError as error:
        raise ValueError('Checkpoint phải nằm trong models/ của repository hiện tại.') from error
    if (lock.get('status') != 'FROZEN' or not identity or identity[0] != 'models'
            or _locked_model_identity(lock.get('checkpoint')) != identity):
        raise ValueError('Bản khóa không trỏ tới checkpoint được yêu cầu.')
    if 'file_sha256' in lock:
        entries = lock['file_sha256']
        expected = {_relative_artifact(name).as_posix(): digest for name, digest in entries.items()}
        expected_count = len(entries)
    else:
        entries = lock['artifact_sha256']
        paths = [(_relative_artifact(name), digest) for name, digest in entries.items()]
        expected = {path.relative_to('best').as_posix(): digest
                    for path, digest in paths if path.parts[0] == 'best'}
        expected_count = sum(path.parts[0] == 'best' for path, _ in paths)
    if len(expected) != expected_count:
        raise ValueError('Bản khóa có artifact trùng sau chuẩn hóa đường dẫn.')
    actual = {}
    for path in sorted(checkpoint.rglob('*')):
        if path.is_file():
            with path.open('rb') as stream:
                actual[path.relative_to(checkpoint).as_posix()] = hashlib.file_digest(stream, 'sha256').hexdigest()
    if not expected or actual != expected:
        raise ValueError('Checkpoint không khớp SHA-256 đã khóa; dừng nạp model.')
    return {'path': str(checkpoint), 'file_sha256': actual, 'verified_against': str(lock_path)}
