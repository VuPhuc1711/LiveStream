"""Bounded, ordered PCM streaming. No label rules and no per-session model loads."""
import asyncio
import hashlib
import json
import logging
import struct
import tempfile
import threading
import time
import wave
import shutil
import anyio
from collections import deque
from pathlib import Path

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from backend import history
from backend.database import ROOT
from backend.speech_buffer import Window, UtteranceBuffer

RATE = 16000
MAX_FRAME_BYTES = RATE * 2 + 4  # at most one second, including sequence uint32 LE
QUEUE_WINDOWS = 3
WINDOW_SECONDS = 12
SILENCE_SECONDS = 1.5
logger = logging.getLogger(__name__)


class ProtocolError(ValueError):
    pass


class ResourceGate:
    """Shared with uploads, in a single local Uvicorn process."""
    def __init__(self):
        self._lock = threading.Lock()

    def claim(self):
        return self._lock.acquire(blocking=False)

    def release(self):
        self._lock.release()


class ChunkOrder:
    def __init__(self):
        self.next = 0
        self.recent = {}
        self.samples = 0

    def receive(self, message):
        if not 6 <= len(message) <= MAX_FRAME_BYTES or (len(message) - 4) % 2:
            raise ProtocolError('Chunk PCM rỗng, sai kích thước hoặc vượt 1 giây.')
        sequence = struct.unpack_from('<I', message)[0]
        digest = hashlib.sha256(message).digest()
        if sequence < self.next:
            if self.recent.get(sequence) != digest:
                raise ProtocolError('Chunk lặp có nội dung khác hoặc đã quá cũ.')
            return sequence, None
        if sequence != self.next:
            raise ProtocolError('Chunk sai thứ tự; kết thúc để tránh thiếu âm thanh.')
        self.recent[sequence] = digest
        if len(self.recent) > 64:
            del self.recent[min(self.recent)]
        self.next += 1
        pcm = message[4:]
        self.samples += len(pcm) // 2
        return sequence, pcm


def process_window(service, window, carry, debug_path=None):
    if not window.pcm:
        return service.analyze_live(None, window.start, True, carry)
    with tempfile.TemporaryDirectory(prefix='livestream_live_') as directory:
        path = Path(directory) / 'window.wav'
        with wave.open(str(path), 'wb') as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(RATE)
            output.writeframes(window.pcm)
        if debug_path is not None:
            shutil.copyfile(path, debug_path)
        return service.analyze_live(path, window.start, window.boundary, carry,
            duration=len(window.pcm)/(RATE*2), voiced_seconds=window.voiced_samples/RATE)


async def monitor(websocket: WebSocket, service, database, gate):
    origin = websocket.headers.get('origin')
    if origin not in {None, 'http://127.0.0.1:5173', 'http://localhost:5173'}:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    if not gate.claim():
        await websocket.send_json({'type': 'error', 'message': 'Tài nguyên đang bận với phiên khác. Hãy thử lại sau.'})
        await websocket.close(code=1013)
        return
    session_id = None
    worker = None
    started = time.perf_counter()
    queue = asyncio.Queue(maxsize=QUEUE_WINDOWS)
    send_lock = asyncio.Lock()
    connected = True
    order = ChunkOrder()
    buffer = None
    failure = ''
    worker_error = ''
    max_pending = 0
    units_count = 0
    warning_count = 0
    asr_unresolved_windows = 0
    asr_recovered_windows = 0
    processing = False
    stop_enqueued = False
    processed_samples = 0
    outstanding_samples = 0
    processing_samples = 0
    unprocessed_samples = 0
    chunk_ranges = deque(maxlen=128)
    accepting_results = True
    debug_directory = None
    debug_recording = None
    debug_vad = None

    async def send(value):
        nonlocal connected
        if not connected:
            return
        try:
            async with send_lock:
                await websocket.send_json({'session_id': session_id, **value})
        except (RuntimeError, WebSocketDisconnect, OSError):
            connected = False

    def metrics():
        silence_samples = buffer.silence_samples if buffer is not None else 0
        assembling = order.samples - silence_samples - processed_samples - outstanding_samples - unprocessed_samples
        return {'pending_windows': max(0, queue.qsize() - int(stop_enqueued)), 'max_pending_windows': max_pending,
                'backlog_seconds': (assembling + outstanding_samples + unprocessed_samples) / RATE,
                'received_seconds': order.samples / RATE,
                'assembling_seconds': assembling / RATE,
                'queued_seconds': (outstanding_samples - processing_samples) / RATE,
                'processing_audio_seconds': processing_samples / RATE,
                'processed_audio_seconds': processed_samples / RATE,
                'silence_seconds': silence_samples / RATE,
                'unprocessed_audio_seconds': unprocessed_samples / RATE,
                'unit_count': units_count, 'warning_count': warning_count, 'processing': processing,
                'asr_unresolved_windows': asr_unresolved_windows,
                'asr_recovered_windows': asr_recovered_windows}

    def abandon():
        nonlocal unprocessed_samples, outstanding_samples, processing_samples, processing
        # Failed/closed sessions have no runnable queue. Do not disguise abandoned audio as completed.
        unprocessed_samples = order.samples - (buffer.silence_samples if buffer is not None else 0) - processed_samples
        outstanding_samples = processing_samples = 0
        processing = False
        while not queue.empty():
            queue.get_nowait()
            queue.task_done()

    def failure_summary():
        counts = metrics()
        return (f" Nhận {counts['received_seconds']:.1f}s; đã xử lý {counts['processed_audio_seconds']:.1f}s; "
                f"không có tiếng nói theo VAD {counts['silence_seconds']:.1f}s; "
                f"chưa xử lý {counts['backlog_seconds']:.1f}s.")

    def register(window):
        nonlocal outstanding_samples
        outstanding_samples += len(window.pcm) // 2
        end = window.start + len(window.pcm) / (RATE*2)
        matches = [seq for seq, start, finish in chunk_ranges if start < end and finish > window.start]
        if matches:
            window.chunk_start, window.chunk_end = matches[0], matches[-1]

    async def enqueue(window):
        nonlocal stop_enqueued
        # A failed inference must not leave producers blocked on a full queue.
        if worker.done():
            return False
        if window is None:
            stop_enqueued = True
        put = asyncio.create_task(queue.put(window))
        await asyncio.wait({put, worker}, return_when=asyncio.FIRST_COMPLETED)
        if not put.done():
            put.cancel()
            await asyncio.gather(put, return_exceptions=True)
            return False
        return True

    async def consume():
        nonlocal units_count, warning_count, worker_error, processing
        nonlocal processed_samples, outstanding_samples, processing_samples
        nonlocal asr_unresolved_windows, asr_recovered_windows
        carry = None
        batch = 0
        try:
            while True:
                window = await queue.get()
                try:
                    if window is None:
                        break
                    if not accepting_results:
                        break
                    processing = True
                    processing_samples = len(window.pcm) // 2
                    await send({'type': 'processing', **metrics()})
                    clock = time.perf_counter()
                    audio_file = f'window_{batch}.wav' if debug_directory and window.pcm else None
                    result = await asyncio.to_thread(process_window, service, window, carry,
                        debug_directory/audio_file if audio_file else None)
                    if not accepting_results:
                        # Inference threads cannot be forcibly cancelled; discard late output.
                        # The captured session_id is never replaced by a subsequent session.
                        processing = False
                        processing_samples = 0
                        break
                    carry = result['pending']
                    quality_warning = result.get('asr_quality_warning')
                    if quality_warning:
                        quality_warning = dict(quality_warning, window_id=f'window_{batch}')
                        asr_unresolved_windows += 1
                    if (result.get('asr_recovery') or {}).get('state') == 'recovered':
                        asr_recovered_windows += 1
                    units = result['units']
                    for unit in units:
                        unit['unit_id'] = f'LIVE_{units_count}'
                        unit['source_segment_id'] = f'window_{batch}'
                        units_count += 1
                        warning_count += unit['final_label'] == 'CANH_BAO'
                    trace = dict(session_id=session_id, window_id=f'window_{batch}', chunk_start=window.chunk_start,
                                                chunk_end=window.chunk_end, start=window.start,
                                                end=window.start+len(window.pcm)/(RATE*2),
                                                sample_rate=RATE, channels=1, samples=len(window.pcm)//2,
                                                pcm_sha256=hashlib.sha256(window.pcm).hexdigest(),
                                                rms=float(np.sqrt(np.mean((np.frombuffer(window.pcm,dtype='<i2').astype(np.float32)/32768)**2))) if window.pcm else 0,
                                                voiced_seconds=window.voiced_samples/RATE,
                                                asr_called=bool(window.pcm), boundary=window.boundary,
                                                debug_audio_file=audio_file,
                                                whisper_raw=result.get('whisper_raw', {}),
                                                asr_rejected_segments=result.get('asr_rejected_segments', []),
                                                asr_recovery=result.get('asr_recovery'),
                                                asr_used_attempt=result.get('asr_used_attempt', 'initial'),
                                                asr_quality_warning=quality_warning,
                                                pending_text=result['temporary_transcript'],
                                                unit_ids=[u['unit_id'] for u in units])
                    if debug_directory:
                        with (debug_directory/'windows.jsonl').open('a', encoding='utf-8') as out:
                            out.write(json.dumps(trace, ensure_ascii=False)+'\n')
                    await asyncio.to_thread(history.append_live_result, database, session_id, units,
                                            result['whisper_count'], time.perf_counter() - started, trace)
                    processed_samples += processing_samples
                    outstanding_samples -= processing_samples
                    processing_samples = 0
                    processing = False
                    await send({'type': 'result', 'classification_units': units,
                        'asr_quality_warning': quality_warning,
                        'window_id': f'window_{batch}', 'chunk_start': window.chunk_start, 'chunk_end': window.chunk_end,
                        'temporary_transcript': result['temporary_transcript'],
                        'processing_time_seconds': time.perf_counter() - clock,
                        'speech_end_seconds': window.speech_end,
                        'latency_seconds': max(0, time.perf_counter() - started - window.speech_end),
                        'final_unit_latencies_seconds': [max(0, time.perf_counter() - started - u['end']) for u in units],
                        'latency_timestamp_is_approximate': True,
                        'context_limit': not window.boundary, **metrics()})
                    batch += 1
                finally:
                    queue.task_done()
        except Exception:
            processing = False
            logger.exception('Live processing failed')
            worker_error = 'Không thể nhận dạng, phân loại hoặc lưu đoạn âm thanh. Phần còn lại chưa được xử lý.'

    try:
        config = await asyncio.wait_for(websocket.receive_json(), timeout=15)
        capture_info = config.pop('capture_info', None) if isinstance(config, dict) else None
        allowed_capture = {'source_sample_rate', 'channel_count', 'auto_gain_control', 'noise_suppression', 'echo_cancellation'}
        if capture_info is not None and (not isinstance(capture_info, dict) or
                set(capture_info)-allowed_capture or any(not isinstance(v, (int, float, bool)) for v in capture_info.values())):
            raise ProtocolError('Thông tin thiết bị thu không hợp lệ.')
        debug_audio = config.pop('debug_audio', False) if isinstance(config, dict) else False
        if not isinstance(debug_audio, bool):
            raise ProtocolError('debug_audio phải là true hoặc false.')
        # Older microphone clients omit source. The source is descriptive only:
        # both inputs share the same PCM validation, resource gate and pipeline.
        source = config.pop('source', 'microphone_live') if isinstance(config, dict) else None
        if not isinstance(source, str) or source not in ('microphone_live', 'browser_tab_live'):
            raise ProtocolError('Nguồn giám sát phải là microphone_live hoặc browser_tab_live.')
        if config != {'type': 'start', 'format': 'pcm_s16le', 'sample_rate': RATE, 'channels': 1}:
            raise ProtocolError('Chỉ nhận PCM 16-bit little-endian, mono, 16000 Hz.')
        health = service.health()
        if not health['ready'] or health.get('classification_mode', 'binary') != 'binary':
            raise ProtocolError('Model hai nhãn chưa sẵn sàng.')
        title = 'Tab livestream' if source == 'browser_tab_live' else 'Microphone trực tiếp'
        session_id = await asyncio.to_thread(history.start_session, database, title, source, health)
        if debug_audio:
            debug_directory = ROOT/'runtime'/'live-debug'/session_id
            debug_directory.mkdir(parents=True, exist_ok=False)
            debug_recording = wave.open(str(debug_directory/'received.wav'), 'wb')
            debug_recording.setnchannels(1)
            debug_recording.setsampwidth(2)
            debug_recording.setframerate(RATE)
            debug_vad = (debug_directory/'vad.jsonl').open('w', encoding='utf-8')
            (debug_directory/'config.json').write_text(json.dumps(dict(session_id=session_id, source=source,
                sample_rate=RATE, channels=1, format='pcm_s16le',
                whisper=health.get('whisper_config', {}).get('model') if health.get('whisper_config') else None,
                language='vi', asr=health.get('whisper_config'),
                checkpoint=health['checkpoint'], vad=health.get('live_asr'), capture_info=capture_info,
                pre_post_roll_seconds=.3, boundary_silence_seconds=1.5),
                ensure_ascii=False, indent=2), encoding='utf-8')
        def trace_vad(frame):
            if debug_vad:
                matches = [seq for seq, start, end in chunk_ranges if start < frame['end'] and end > frame['start']]
                debug_vad.write(json.dumps(dict(session_id=session_id,
                    chunk_start=matches[0] if matches else None, chunk_end=matches[-1] if matches else None,
                    sample_rate=RATE, **frame))+'\n')
        # One startup-loaded model, with recurrent state reset for each isolated session.
        buffer = UtteranceBuffer(max_seconds=WINDOW_SECONDS, vad=service.live_vad, trace=trace_vad if debug_vad else None)
        if debug_directory:
            (debug_directory/'windows.jsonl').touch()  # Empty means no ASR was called.
        await send({'type': 'started', 'session_id': session_id, 'source': source, 'model_version': health['model_version'],
                    'debug_directory': debug_directory.relative_to(ROOT).as_posix() if debug_directory else None,
                    'checkpoint': health['checkpoint'], 'max_chunk_bytes': MAX_FRAME_BYTES,
                    'sample_rate': RATE, 'queue_limit': QUEUE_WINDOWS})
        worker = asyncio.create_task(consume())
        overflow = []
        while True:
            if worker_error:
                failure = worker_error
                break
            message = await asyncio.wait_for(websocket.receive(), timeout=60)
            if message['type'] == 'websocket.disconnect':
                connected = False
                accepting_results = False
                failure = 'Phiên bị gián đoạn do mất kết nối; chỉ kết quả đã hoàn thành được giữ.'
                break
            if message.get('bytes') is not None:
                sequence, pcm = order.receive(message['bytes'])
                if pcm is not None:
                    if debug_recording:
                        debug_recording.writeframesraw(pcm)
                        with (debug_directory/'chunks.jsonl').open('a', encoding='utf-8') as out:
                            out.write(json.dumps(dict(session_id=session_id, chunk_id=sequence,
                                start=(order.samples-len(pcm)//2)/RATE, end=order.samples/RATE,
                                sample_rate=RATE, samples=len(pcm)//2,
                                pcm_sha256=hashlib.sha256(pcm).hexdigest()))+'\n')
                    chunk_ranges.append((sequence, (order.samples-len(pcm)//2)/RATE, order.samples/RATE))
                    for window in buffer.feed(pcm):
                        register(window)
                        if queue.full():
                            overflow.append(window)
                        else:
                            queue.put_nowait(window)
                            max_pending = max(max_pending, queue.qsize())
                await send({'type': 'ack', 'sequence': sequence, 'duplicate': pcm is None, **metrics()})
                if overflow:
                    failure = 'Backend quá tải: đã dừng nhận audio. Sẽ xử lý hết phần đã nhận; phần nói sau mốc nhận cuối chưa được gửi/xử lý.'
                    break
                if debug_recording and order.samples*2 >= 50*1024*1024:
                    failure = 'Đã đạt giới hạn audio debug 50 MiB; kết thúc và xử lý phần đã nhận.'
                    break
            else:
                try:
                    command = json.loads(message.get('text') or '')
                except ValueError:
                    raise ProtocolError('Lệnh WebSocket không hợp lệ.') from None
                if command.get('type') != 'stop' or command.get('last_sequence') != order.next - 1:
                    raise ProtocolError('Lệnh kết thúc không khớp chunk cuối; không thể bảo đảm đủ audio.')
                break
        await send({'type': 'draining', 'message': failure or 'Đang xử lý phần còn lại', **metrics()})
        # Accepted audio is bounded, including one overflow window and final remainder.
        remaining = buffer.finish()
        for window in remaining:
            register(window)
        for window in [*overflow, *remaining]:
            if window is not None and not worker.done():
                await enqueue(window)
                max_pending = max(max_pending, queue.qsize())
        if not worker.done():
            await enqueue(None)
        await worker
        if worker_error:
            failure = worker_error
        if failure:
            abandon()
            await asyncio.to_thread(history.fail_session, database, session_id,
                failure + failure_summary(), time.perf_counter() - started)
            await send({'type': 'error', 'session_id': session_id, 'message': failure, **metrics()})
        else:
            if metrics()['backlog_seconds'] != 0 or not queue.empty() or processing:
                raise RuntimeError('Accepted audio has not been fully accounted for')
            await asyncio.to_thread(history.finish_live_session, database, session_id, time.perf_counter() - started)
            await send({'type': 'completed', 'session_id': session_id, **metrics()})
    except (ProtocolError, asyncio.TimeoutError, WebSocketDisconnect) as error:
        accepting_results = False
        abandon()
        failure = str(error) if isinstance(error, ProtocolError) else 'Kết nối bị gián đoạn hoặc không nhận thêm audio trong 60 giây.'
        await send({'type': 'error', 'message': failure, 'session_id': session_id, **metrics()})
    except Exception:
        accepting_results = False
        abandon()
        logger.exception('Live session failed')
        failure = 'Không thể mở hoặc lưu phiên giám sát. Vui lòng kiểm tra backend.'
        await send({'type': 'error', 'message': failure, 'session_id': session_id, **metrics()})
    except asyncio.CancelledError:
        accepting_results = False
        abandon()
        connected = False
        failure = 'Phiên bị gián đoạn khi kết nối hoặc backend dừng; phần chưa chốt chưa được xử lý.'
        raise
    finally:
        # ASGI cancellation must not skip persistence or leave the resource gate held.
        with anyio.CancelScope(shield=True):
            try:
                if worker and not worker.done():
                    await enqueue(None)
                    try:
                        await worker
                    except asyncio.CancelledError:
                        failure = failure or 'Phiên bị gián đoạn; phần chưa chốt chưa được xử lý.'
                if failure and session_id:
                    try:
                        await asyncio.to_thread(history.fail_session, database, session_id,
                            failure + failure_summary(), time.perf_counter() - started)
                    except Exception:
                        logger.exception('Could not mark interrupted live session')
            finally:
                try:
                    if debug_recording:
                        debug_recording.close()
                    if debug_vad:
                        debug_vad.close()
                    if debug_directory:
                        (debug_directory/'completion.json').write_text(json.dumps(
                            dict(session_id=session_id, status='failed' if failure else 'completed', error=failure,
                                 **metrics()), ensure_ascii=False, indent=2), encoding='utf-8')
                except OSError:
                    logger.exception('Could not finalize optional audio debug files')
                finally:
                    while not queue.empty():
                        queue.get_nowait()
                    if buffer is not None:
                        buffer.buffer.clear()
                        buffer.pre.clear()
                        buffer.remainder.clear()
                    gate.release()
            if connected:
                try:
                    await websocket.close()
                except (RuntimeError, OSError):
                    pass
