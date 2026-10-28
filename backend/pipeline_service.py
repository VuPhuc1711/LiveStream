"""Own models for one application lifespan; reuse existing inference logic."""
import logging
import threading
import hashlib
import tempfile
import wave
from pathlib import Path
from collections import Counter

from backend.model_runtime import check_environment, create_detector, verify_checkpoint
from backend.model_config import ROOT, get_active_model
from nlp.label_mapping import labels_for_mode


class InvalidAudioError(ValueError):
    pass


class PipelineService:
    def __init__(self):
        self.checkpoint = None
        self.model_version = None
        self.phobert_device = None
        self.transcriber = None
        self.asr_configuration = None
        self.live_vad = None
        self.detector = None
        self.classification_mode = 'three_class'
        self.labels = labels_for_mode(self.classification_mode)
        self.lock = threading.RLock()
        self.startup_fault = None

    def start(self):
        from speech.transcribe import WhisperTranscriber
        from speech.asr_config import load_asr_config
        from backend.neural_vad import NeuralVad
        self.live_vad = NeuralVad()
        environment = check_environment()
        # Resolve version and path from one startup snapshot, never from live config in health().
        active = get_active_model()
        if active.get('classifier', 'phobert_only') != 'phobert_only':
            raise ValueError('Backend chỉ hỗ trợ chế độ phobert_only.')
        checkpoint = (ROOT / active['checkpoint']).resolve()
        self.classification_mode = active.get('classification_mode', 'three_class')
        self.labels = labels_for_mode(self.classification_mode)
        verify_checkpoint(checkpoint)
        logging.getLogger(__name__).info('Verified checkpoint: %s', checkpoint)
        asr = load_asr_config()
        self.transcriber = WhisperTranscriber(model_name=asr['model'], device=asr['device'],
            precision=asr['precision'], initial_prompt=asr['initial_prompt'],
            language=asr['language'], task=asr['task'], download_root=asr['download_root'])
        device = active.get('backend_device', 'auto')
        if device not in {'auto', 'cpu', 'cuda'}:
            raise ValueError('backend_device must be auto, cpu or cuda')
        if device == 'auto':
            device = 'cuda' if environment['cuda_available'] else 'cpu'
        if self.classification_mode == 'three_class':
            self.detector = create_detector(checkpoint, device)
        else:
            self.detector = create_detector(checkpoint, device, self.classification_mode)
        loaded = self.detector.phobert_detector.checkpoint_path.resolve()
        if loaded != checkpoint:
            raise RuntimeError('Loaded checkpoint does not match startup configuration')
        self.checkpoint = loaded.relative_to(ROOT).as_posix() if loaded.is_relative_to(ROOT) else str(loaded)
        self.model_version = active['version']
        self.phobert_device = str(self.detector.phobert_detector.device)
        # A model merely loaded into VRAM may still fail its first worker GEMM.
        # Check after PhoBERT is resident, under the actual combined memory load.
        try:
            self.transcriber.verify_runtime()
        except Exception:
            self.startup_fault = 'asr_runtime_self_check_failed'
            raise
        self.asr_configuration = self.transcriber.configuration()
        logging.getLogger('uvicorn.error').info('Loaded Whisper ASR configuration: %s', self.asr_configuration)
        logging.getLogger('uvicorn.error').info('Loaded PhoBERT %s checkpoint: %s (device=%s)', self.model_version, loaded, self.phobert_device)
        logging.getLogger('uvicorn.error').info('Live ASR speech admission: %s', self.live_vad.configuration())

    def health(self):
        from backend.upload_audio import WINDOW_SECONDS, SHORT_TAIL_SECONDS
        from backend.asr_quality import guard_configuration
        runtime_fault = getattr(self.transcriber, 'runtime_fault', None)
        runtime_fault = runtime_fault if isinstance(runtime_fault, dict) else None
        ready = self.transcriber is not None and self.detector is not None and not runtime_fault and not self.startup_fault
        return {'status': 'ok' if ready else 'not_ready', 'checkpoint': self.checkpoint,
                'model_version': self.model_version,
                'classification_mode': self.classification_mode, 'labels': list(self.labels),
                'classifier': 'phobert_only',
                'phobert_device': self.phobert_device,
                'whisper_config': self.asr_configuration,
                'asr_runtime': runtime_fault or {'status': 'available' if ready else 'not_ready',
                                                'startup_fault': self.startup_fault},
                'live_asr': self.live_vad.configuration() if self.live_vad is not None else None,
                'asr_quality_guard': guard_configuration(),
                'upload_asr': {'speech_admission': 'shared_silero', 'window_seconds': WINDOW_SECONDS,
                               'pre_post_roll_seconds': .3, 'short_contiguous_tail_merge_seconds': SHORT_TAIL_SECONDS},
                'whisper': 'ready' if self.transcriber is not None and not runtime_fault and not self.startup_fault else 'not_ready',
                'phobert': 'ready' if self.detector is not None else 'not_ready', 'ready': ready}

    def analyze(self, path):
        from backend.upload_audio import decoded_pcm, UploadSpeechWindows, AudioDecodeError
        from backend.speech_buffer import RATE
        with self.lock:
            if not self.health()['ready']:
                raise RuntimeError('Pipeline is not ready')
            units, traces, carry = [], [], None
            whisper_count = asr_samples = 0
            source = UploadSpeechWindows(self.live_vad)
            try:
                with decoded_pcm(path) as chunks, tempfile.TemporaryDirectory(prefix='upload_speech_') as directory:
                    for index, window in enumerate(source.windows(chunks)):
                        temporary = Path(directory) / 'speech.wav'
                        if window.pcm:
                            with wave.open(str(temporary), 'wb') as output:
                                output.setnchannels(1); output.setsampwidth(2); output.setframerate(RATE)
                                output.writeframes(window.pcm)
                        result = self.analyze_live(temporary if window.pcm else None, window.start,
                            window.boundary, carry, len(window.pcm)/(RATE*2), window.voiced_samples/RATE)
                        carry = result['pending']
                        for unit in result['units']:
                            unit['unit_id'] = f'U{len(units)+1:04d}'
                            unit['source_segment_id'] = f'upload_window_{index}'
                            units.append(unit)
                        whisper_count += result['whisper_count']
                        asr_samples += len(window.pcm)//2
                        traces.append(dict(window_id=f'upload_window_{index}', start=window.start,
                            end=window.start+len(window.pcm)/(RATE*2), sample_rate=RATE, channels=1,
                            samples=len(window.pcm)//2, pcm_sha256=hashlib.sha256(window.pcm).hexdigest(),
                            voiced_seconds=window.voiced_samples/RATE, asr_called=bool(window.pcm),
                            boundary=window.boundary, asr_config=self.asr_configuration,
                            whisper_raw=result['whisper_raw'],
                            asr_rejected_segments=result['asr_rejected_segments'],
                            asr_recovery=result.get('asr_recovery'),
                            asr_used_attempt=result.get('asr_used_attempt', 'initial'),
                            asr_quality_warning=dict(result['asr_quality_warning'], window_id=f'upload_window_{index}')
                                if result.get('asr_quality_warning') else None,
                            pending_text=result['temporary_transcript'], unit_ids=[u['unit_id'] for u in result['units']]))
            except AudioDecodeError as error:
                raise InvalidAudioError(str(error)) from error
            finally:
                self.live_vad.reset()
            if carry or asr_samples + source.buffer.silence_samples != source.received_samples:
                raise RuntimeError('Upload audio/text has not been fully accounted for')
            counts = Counter(unit['final_label'] for unit in units)
            if any(u['final_label'] not in self.labels or
                   set(u['phobert_result']['probabilities']) != set(self.labels) for u in units):
                raise RuntimeError('Inference labels do not match configured task')
            return {'classification_mode': self.classification_mode,
                    'transcript': ' '.join(unit['text'] for unit in units), 'classification_units': units,
                    'asr_windows': traces,
                    'asr_quality_warnings':[t['asr_quality_warning'] for t in traces if t.get('asr_quality_warning')],
                    'asr_unresolved_windows':sum(bool(t.get('asr_quality_warning')) for t in traces),
                    'asr_recovered_windows':sum((t.get('asr_recovery') or {}).get('state')=='recovered' for t in traces),
                    'audio_accounting': dict(received_seconds=source.received_samples/RATE,
                        processed_audio_seconds=asr_samples/RATE,
                        silence_seconds=source.buffer.silence_samples/RATE, unprocessed_audio_seconds=0.),
                    'summary': {'whisper_segment_count': whisper_count,
                                'classification_unit_count': len(units),
                                'final_label_counts': {label: counts[label] for label in
                                    self.labels}}}

    def close(self):
        self.live_vad = None
        self.asr_configuration = None
        self.detector = None
        self.checkpoint = None
        self.model_version = None
        self.phobert_device = None
        self.startup_fault = None
        if self.transcriber is not None:
            self.transcriber.close()
            self.transcriber = None

    def analyze_live(self, path, offset, boundary, carry=None, duration=None, voiced_seconds=None):
        """Same Whisper options/tokenizer/classifier; defer the last sentence at a
        forced audio cut, then join it to the next window before classification.
        Audio itself never overlaps, so genuine repeated speech is not deduplicated.
        """
        from speech.sentence_units import build_classification_units
        from backend.asr_quality import timestamp_issues, raw_snapshot, same_decoded_text, REVISION
        with self.lock:
            raw = self.transcriber.transcribe(str(path)) if path is not None else {'segments': []}
            original = raw
            issues = timestamp_issues(original, duration)
            rejected = [dict(item, attempt='initial') for item in issues]
            recovery = None
            quality_warning = None
            used_attempt = 'initial'
            if path is not None and issues:
                # Retry the SAME full audio, changing only initial_prompt=None.
                # Native timestamps are retained. Alignment and split retries
                # failed real trials and must not rehabilitate an invalid decode.
                recovery = dict(revision=REVISION, state='unresolved', trigger=issues,
                    reason='invalid_audio_timestamp', retried=True,
                    strategy='full_window_no_prompt_v1',
                    configuration=(self.asr_configuration or {}).get('recovery', {}))
                weak_tail = (duration is not None and duration <= .5 and
                    voiced_seconds is not None and voiced_seconds < .1)
                if weak_tail:
                    # Do not solicit another caption from a weak VAD hangover.
                    # Valid short words never enter this invalid-decode branch.
                    recovery.update(reason='invalid_decode_on_weak_short_tail', retried=False)
                else:
                    try:
                        retry = self.transcriber.transcribe_without_prompt(str(path))
                        retry_issues = timestamp_issues(retry, duration)
                        usable = any(any(c.isalnum() for c in str(s.get('text', '')))
                            for s in retry.get('segments', []))
                        recovery.update(whisper_raw=raw_snapshot(retry), issues=retry_issues)
                        if usable and not retry_issues and not same_decoded_text(original, retry):
                            raw = retry
                            recovery.update(state='recovered', reason='native_decode_without_prompt')
                            used_attempt = 'without_prompt'
                        else:
                            recovery['reason'] = ('retry_timestamps_invalid' if retry_issues else
                                'retry_unchanged_text' if usable else 'retry_no_usable_text')
                    except Exception as error:
                        logging.getLogger(__name__).exception('Bounded ASR recovery failed')
                        recovery.update(reason='retry_failed', error_type=type(error).__name__)
                if recovery['state'] != 'recovered':
                    # Never bridge an unknown interval: the preceding incomplete
                    # clause might be missing its negation or comparison scope.
                    quality_warning = dict(start=carry['start'] if carry else offset,
                        end=offset+(duration or 0), reason=recovery['reason'], retried=recovery['retried'])
                    if carry:
                        recovery['uncommitted_context'] = dict(carry)
                    carry = None
                    raw = {'segments': []}
                    used_attempt = None
                logging.getLogger(__name__).warning(
                    'ASR timestamp admission offset=%.3f duration=%s state=%s reason=%s',
                    offset, duration, recovery['state'], recovery['reason'])
            accepted = []
            for segment in raw['segments']:
                if duration is not None and (segment.get('start', 0) >= duration or
                                             segment.get('end', 0) <= segment.get('start', 0)):
                    rejected.append({'segment_id': segment.get('id'), 'reason': 'no_timestamp_overlap_with_received_audio'})
                    continue
                accepted.append(segment)
            units = build_classification_units(accepted)
            for unit in units:
                # Only minor frame-rounding reaches this clamp. Severe geometry
                # is rejected above before text can become carry or a prediction.
                if duration is not None:
                    unit['timestamp_clamped'] = unit['end'] > duration or unit['start'] > duration
                    unit['start'] = min(unit['start'], duration)
                    unit['end'] = min(unit['end'], duration)
                unit['start'] += offset
                unit['end'] += offset
                unit['timestamp_is_approximate'] = True
            if carry:
                if units:
                    units[0]['text'] = carry['text'] + ' ' + units[0]['text']
                    units[0]['start'] = carry['start']
                else:
                    units = [carry]
            pending = None
            # Keep a boundary sentence even when ASR inserted punctuation there.
            # Bound context retention to 24 seconds rather than waiting forever.
            if not boundary and units and units[-1]['end'] - units[-1]['start'] < 24:
                pending = units.pop()
            for unit in units:
                unit.update(self.detector.predict(unit['text']))
            return {'units': units, 'pending': pending, 'temporary_transcript': pending['text'] if pending else '',
                    'whisper_count': len(raw['segments']),
                    'asr_rejected_segments': rejected,
                    'asr_recovery': recovery, 'asr_used_attempt': used_attempt,
                    'asr_quality_warning': quality_warning,
                    # Keep traceable raw text/timestamps/scores, never Whisper token arrays.
                    'whisper_raw': raw_snapshot(original)}
