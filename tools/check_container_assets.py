"""Offline preflight for separately supplied weights; stdlib only, no downloads."""
import hashlib
import json
from pathlib import Path

from backend.model_config import ROOT, get_active_model, verify_model_checkpoint


def check():
    assets = json.loads((ROOT / 'config/runtime_assets.json').read_text(encoding='utf-8'))
    active = get_active_model()
    if active['checkpoint'] != assets['phobert']['checkpoint'] or active['version'] != 'binary_v04':
        raise ValueError('Asset handoff manifest does not match the selected binary_v04 checkpoint')
    checkpoint = ROOT / assets['phobert']['checkpoint']
    lock = checkpoint.parent / 'checkpoint_lock.json'
    if hashlib.sha256(lock.read_bytes()).hexdigest() != assets['phobert']['lock_sha256']:
        raise ValueError('Frozen checkpoint lock differs from the shared handoff manifest')
    verified = verify_model_checkpoint(checkpoint)
    whisper = ROOT / assets['whisper']['path']
    with whisper.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != assets['whisper']['sha256']:
        raise ValueError('Turbo cache does not match its official SHA-256')
    return {'phobert': active['version'], 'artifact_count': len(verified['file_sha256']),
            'frozen_lock_unchanged': True, 'whisper': assets['whisper']['model'],
            'whisper_sha256_matches': True, 'downloaded': False, 'model_loaded': False}


if __name__ == '__main__':
    print(json.dumps(check(), ensure_ascii=False, indent=2))
