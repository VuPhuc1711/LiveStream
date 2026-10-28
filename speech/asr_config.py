"""One immutable startup snapshot, separate from the PhoBERT checkpoint config."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VOCABULARY_PROMPT = 'TikTok, TikTok Shop, Shopee, Lazada, Zalo, Messenger, top 1, lượt bán, gian hàng.'
DEFAULTS = {'model': 'small', 'device': 'cpu', 'precision': 'fp32',
            'language': 'vi', 'task': 'transcribe', 'initial_prompt': '', 'download_root': None}


def load_asr_config(path=None):
    location = Path(path or os.environ.get('WHISPER_CONFIG_PATH', ROOT / 'asr_config.json'))
    if not location.is_absolute():
        location = ROOT / location
    config = dict(DEFAULTS)
    if location.is_file():
        values = json.loads(location.read_text(encoding='utf-8'))
        if not isinstance(values, dict) or set(values) - set(DEFAULTS):
            raise ValueError('Cấu hình ASR có trường không hợp lệ.')
        config.update(values)
    elif path or os.environ.get('WHISPER_CONFIG_PATH'):
        raise FileNotFoundError('Không tìm thấy file cấu hình ASR được chỉ định.')
    if config['model'] not in {'tiny', 'base', 'small', 'medium', 'large', 'large-v1', 'large-v2', 'large-v3', 'large-v3-turbo', 'turbo'}:
        raise ValueError('ASR phải dùng model Whisper đa ngôn ngữ.')
    if config['device'] not in {'cpu', 'cuda'} or config['precision'] not in {'fp32', 'fp16'}:
        raise ValueError('Thiết bị/chế độ số học ASR không hợp lệ.')
    if config['device'] == 'cpu' and config['precision'] != 'fp32':
        raise ValueError('Whisper CPU phải dùng fp32.')
    if config['language'] != 'vi' or config['task'] != 'transcribe':
        raise ValueError('Luồng tiếng Việt dùng language=vi, task=transcribe.')
    if not isinstance(config['initial_prompt'], str) or len(config['initial_prompt']) > 300:
        raise ValueError('initial_prompt chỉ dành cho gợi ý từ vựng ngắn (tối đa 300 ký tự).')
    if config['download_root'] is not None:
        if not isinstance(config['download_root'], str):
            raise ValueError('download_root phải là đường dẫn.')
        cache = Path(config['download_root']).expanduser()
        config['download_root'] = str(cache if cache.is_absolute() else ROOT / cache)
    return config
