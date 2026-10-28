import base64
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from speech.asr_config import DEFAULTS, VOCABULARY_PROMPT, load_asr_config
from speech.transcribe import WhisperTranscriber, load_cached_half_precision, stage_half_precision


class AsrConfigTests(unittest.TestCase):
    def test_cuda_execution_failure_is_reported_separately_from_startup_snapshot(self):
        with tempfile.TemporaryDirectory() as directory, patch('speech.transcribe.load_cached_half_precision') as load, \
             patch('speech.transcribe.shutil.which', return_value='ffmpeg'):
            path=Path(directory)/'audio.wav';path.write_bytes(b'mock_not_audio')
            transcriber=WhisperTranscriber('turbo',device='cuda',precision='fp16')
            snapshot=transcriber.configuration()
            load.return_value.transcribe.side_effect=[RuntimeError('CUDA CUBLAS_STATUS_NOT_SUPPORTED'),{'segments':[]}]
            with self.assertRaisesRegex(RuntimeError,'CUBLAS'):
                transcriber.transcribe(path)
            self.assertEqual(transcriber.runtime_fault['category'],'cuda_execution_error')
            self.assertEqual(snapshot['runtime_self_check'],{'status':'not_run'})
            transcriber.transcribe(path)
            self.assertIsNone(transcriber.runtime_fault)

    def test_default_matches_existing_cpu_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'asr.json'
            path.write_text('{}', encoding='utf-8')
            self.assertEqual(load_asr_config(path), DEFAULTS)

    def test_reject_invalid_language_model_precision_and_full_transcript_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'asr.json'
            for values in ({'model':'small.en'}, {'language':'en'}, {'task':'translate'},
                           {'precision':'fp16'}, {'initial_prompt':'x'*301}, {'unknown':1}):
                path.write_text(json.dumps(values), encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_asr_config(path)

    def test_explicit_missing_config_does_not_silently_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_asr_config(Path(directory) / 'absent.json')

    def test_prompt_applies_each_call_without_previous_session_text_or_reload(self):
        with tempfile.TemporaryDirectory() as directory, patch('speech.transcribe.whisper.load_model') as load, \
             patch('speech.transcribe.shutil.which', return_value='ffmpeg'):
            path = Path(directory) / 'audio.wav'; path.write_bytes(b'not_decoded_by_mock')
            load.return_value.transcribe.side_effect = [dict(text='Phiên trước.', segments=[]), dict(text='Không.', segments=[])]
            transcriber = WhisperTranscriber('small', device='cpu', initial_prompt=VOCABULARY_PROMPT)
            transcriber.transcribe(path); transcriber.transcribe(path)
            load.assert_called_once()
            for call in load.return_value.transcribe.call_args_list:
                self.assertEqual(call.kwargs['initial_prompt'], VOCABULARY_PROMPT)
                self.assertEqual(call.kwargs['language'], 'vi')
                self.assertEqual(call.kwargs['task'], 'transcribe')
                self.assertFalse(call.kwargs['fp16'])
                self.assertFalse(call.kwargs['carry_initial_prompt'])
            self.assertEqual(transcriber.configuration()['model'], 'small')
            transcriber.close()
            with self.assertRaises(RuntimeError): transcriber.transcribe(path)

    def test_fp16_staged_on_cpu_then_moved_to_gpu(self):
        with patch('speech.transcribe.load_cached_half_precision') as load:
            transcriber = WhisperTranscriber('turbo', device='cuda', precision='fp16', download_root='cache')
            load.assert_called_once_with('turbo', 'cache')
            load.return_value.half.assert_called_once()
            self.assertEqual(str(load.return_value.to.call_args.args[0]), 'cuda')
            self.assertTrue(transcriber.decode_options['fp16'])
            self.assertEqual(transcriber.configuration()['loading_strategy'], 'verified_cached_mmap_fp16_assign_v1')

    def test_explicit_retry_reuses_model_and_preserves_raw_alignment_and_normal_options(self):
        with tempfile.TemporaryDirectory() as directory, patch('speech.transcribe.load_cached_half_precision') as load, \
             patch('speech.transcribe.shutil.which', return_value='ffmpeg'):
            path = Path(directory) / 'audio.wav'; path.write_bytes(b'not_decoded_by_mock')
            raw = dict(text='Không.', segments=[dict(id=0, start=0., end=.3, text='Không.',
                words=[dict(word='Không.', start=0., end=.3, probability=.9)])])
            load.return_value.transcribe.return_value = raw
            transcriber = WhisperTranscriber('turbo', device='cuda', precision='fp16',
                initial_prompt=VOCABULARY_PROMPT)
            original = dict(transcriber.decode_options)
            transcriber.transcribe(path)
            self.assertIs(transcriber.transcribe_retry(path), raw)
            transcriber.transcribe(path)
            load.assert_called_once()
            calls = load.return_value.transcribe.call_args_list
            self.assertEqual(len(calls), 3)
            self.assertEqual(calls[0].kwargs, original)
            self.assertEqual(calls[2].kwargs, original)
            self.assertEqual(transcriber.decode_options, original)
            retry = calls[1].kwargs
            self.assertEqual(calls[1].args, (str(path.resolve()),))
            self.assertEqual(retry, dict(original, temperature=0., beam_size=5,
                condition_on_previous_text=False, word_timestamps=True))
            self.assertEqual(retry['language'], 'vi')
            self.assertEqual(retry['task'], 'transcribe')
            self.assertTrue(retry['fp16'])
            self.assertEqual(retry['initial_prompt'], VOCABULARY_PROMPT)
            self.assertFalse(retry['carry_initial_prompt'])
            self.assertNotIn('hallucination_silence_threshold', retry)
            recovery = transcriber.configuration()['diagnostic_retry']
            self.assertFalse(recovery['enabled_in_pipeline'])
            self.assertEqual(recovery['revision'], 'bounded_timestamp_retry_trial_v1')
            self.assertEqual(recovery['decoding'], retry)
            recovery['decoding']['temperature'] = .8
            self.assertEqual(transcriber.retry_decode_options['temperature'], 0.)

    def test_retry_uses_same_path_environment_and_released_model_checks(self):
        with tempfile.TemporaryDirectory() as directory, patch('speech.transcribe.whisper.load_model') as load, \
             patch('speech.transcribe.shutil.which', return_value='ffmpeg') as ffmpeg:
            transcriber = WhisperTranscriber('small', device='cpu')
            missing = Path(directory) / 'missing.wav'
            for method in (transcriber.transcribe, transcriber.transcribe_retry):
                with self.assertRaises(FileNotFoundError): method(missing)
            path = Path(directory) / 'audio.wav'; path.write_bytes(b'not_decoded_by_mock')
            ffmpeg.return_value = None
            for method in (transcriber.transcribe, transcriber.transcribe_retry):
                with self.assertRaisesRegex(RuntimeError, 'FFmpeg'): method(path)
            transcriber.close()
            for method in (transcriber.transcribe, transcriber.transcribe_retry):
                with self.assertRaises(RuntimeError): method(path)
            load.return_value.transcribe.assert_not_called()

    def test_production_recovery_changes_only_prompt_and_does_not_contaminate_next_call(self):
        with tempfile.TemporaryDirectory() as directory, patch('speech.transcribe.whisper.load_model') as load, \
             patch('speech.transcribe.shutil.which', return_value='ffmpeg'):
            path=Path(directory)/'window.wav';path.write_bytes(b'mock_not_audio')
            transcriber=WhisperTranscriber('small',device='cpu',initial_prompt=VOCABULARY_PROMPT)
            original=dict(transcriber.decode_options)
            transcriber.transcribe(path)
            transcriber.transcribe_without_prompt(path)
            transcriber.transcribe(path)
            calls=load.return_value.transcribe.call_args_list
            self.assertEqual(calls[0].kwargs,original)
            self.assertEqual(calls[1].kwargs,dict(original,initial_prompt=None))
            self.assertEqual(calls[2].kwargs,original)
            self.assertFalse(calls[1].kwargs['word_timestamps'])
            load.assert_called_once()

    def test_legacy_cli_gpu_default_preserves_float_weights(self):
        with patch('speech.transcribe.whisper.load_model') as load:
            transcriber = WhisperTranscriber('small', device='cuda')
            self.assertEqual(str(load.call_args.kwargs['device']), 'cuda')
            load.return_value.half.assert_not_called()
            self.assertTrue(transcriber.decode_options['fp16'])
            self.assertEqual(transcriber.configuration()['weights_precision'], 'fp32')

    def test_verified_mmap_half_loader_matches_legacy_staging_including_layer_norm_bits(self):
        import torch
        from whisper.model import Whisper, ModelDimensions
        dims = ModelDimensions(n_mels=80, n_audio_ctx=2, n_audio_state=4,
            n_audio_head=2, n_audio_layer=1, n_vocab=100, n_text_ctx=4,
            n_text_state=4, n_text_head=2, n_text_layer=1)
        torch.manual_seed(42)
        source = Whisper(dims)
        with torch.no_grad():
            for parameter in source.parameters(): parameter.uniform_(-.1, .1)
        state = {name: value.half() for name, value in source.state_dict().items()}
        dump = base64.b85encode(gzip.compress(bytes([1, 0])))
        expected = Whisper(dims)
        expected.load_state_dict(state)
        expected.set_alignment_heads(dump)
        stage_half_precision(expected)
        original_dtype = torch.get_default_dtype()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test-model.pt'
            torch.save(dict(dims=vars(dims), model_state_dict=state), path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.dict('speech.transcribe.whisper._MODELS', {'test-model': f'https://example.invalid/{digest}/test-model.pt'}), \
                 patch.dict('speech.transcribe.whisper._ALIGNMENT_HEADS', {'test-model': dump}), \
                 patch('speech.transcribe.whisper.load_model') as official:
                actual = load_cached_half_precision('test-model', directory)
                stage_half_precision(actual)
                official.assert_not_called()
                self.assertEqual(torch.get_default_dtype(), original_dtype)
                self.assertEqual(set(actual.state_dict()), set(expected.state_dict()))
                for name, value in actual.state_dict().items():
                    self.assertEqual(value.dtype, expected.state_dict()[name].dtype, name)
                    self.assertTrue(torch.equal(value, expected.state_dict()[name]), name)
                actual_buffers, expected_buffers = dict(actual.named_buffers()), dict(expected.named_buffers())
                self.assertEqual(set(actual_buffers), set(expected_buffers))
                for name, value in actual_buffers.items():
                    other = expected_buffers[name]
                    self.assertFalse(value.is_meta, name)
                    self.assertEqual(value.dtype, other.dtype, name)
                    self.assertEqual(value.layout, other.layout, name)
                    self.assertTrue(torch.equal(value.to_dense() if value.is_sparse else value,
                        other.to_dense() if other.is_sparse else other), name)
                self.assertFalse(any(value.is_meta for value in actual.parameters()))
                self.assertEqual(actual.encoder.ln_post.weight.dtype, torch.float32)
                mel = torch.randn(1, 80, 4, dtype=torch.float16)
                tokens = torch.tensor([[1, 2]])
                with torch.inference_mode():
                    expected_output, actual_output = expected(mel, tokens), actual(mel, tokens)
                self.assertTrue(torch.isfinite(actual_output).all())
                self.assertTrue(torch.equal(actual_output, expected_output))
                with patch('speech.transcribe.whisper.model.AudioEncoder', side_effect=RuntimeError('constructor failed')):
                    with self.assertRaisesRegex(RuntimeError, 'constructor failed'):
                        load_cached_half_precision('test-model', directory)
                self.assertEqual(torch.get_default_dtype(), original_dtype)
                self.assertEqual(torch.empty(0).device.type, 'cpu')

    def test_meta_assembly_uses_no_parameter_storage_and_does_not_change_default_dtype(self):
        import torch
        from speech.transcribe import _whisper_without_parameter_storage
        from whisper.model import ModelDimensions
        dims = ModelDimensions(80, 2, 4, 2, 1, 100, 4, 4, 2, 1)
        dtype, device = torch.get_default_dtype(), torch.empty(0).device
        with patch('speech.transcribe.torch.set_default_dtype') as set_dtype:
            model = _whisper_without_parameter_storage(dims)
            self.assertTrue(all(value.is_meta for value in model.parameters()))
            self.assertTrue(model.encoder.positional_embedding.is_meta)
            self.assertEqual(model.decoder.mask.device.type, 'cpu')
            self.assertEqual(model.decoder.mask.shape, (4, 4))
            set_dtype.assert_not_called()
        self.assertEqual(torch.get_default_dtype(), dtype)
        self.assertEqual(torch.empty(0).device, device)

    def test_low_memory_loader_rejects_missing_or_modified_cache_before_loading(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict('speech.transcribe.whisper._MODELS', {'test-model': f'https://example.invalid/{"0"*64}/test-model.pt'}), \
             patch.dict('speech.transcribe.whisper._ALIGNMENT_HEADS', {'test-model': b'not_used'}), \
             patch('speech.transcribe.torch.load') as load:
            with self.assertRaises(FileNotFoundError): load_cached_half_precision('test-model', directory)
            (Path(directory) / 'test-model.pt').write_bytes(b'modified')
            with self.assertRaisesRegex(RuntimeError, 'SHA256'): load_cached_half_precision('test-model', directory)
            load.assert_not_called()

    def test_whisper_half_keeps_float_layer_norm_and_runs_real_forward(self):
        import torch
        from whisper.model import Whisper, ModelDimensions
        model = Whisper(ModelDimensions(n_mels=80, n_audio_ctx=2, n_audio_state=4,
            n_audio_head=2, n_audio_layer=1, n_vocab=100, n_text_ctx=4,
            n_text_state=4, n_text_head=2, n_text_layer=1))
        stage_half_precision(model)
        self.assertEqual(model.encoder.ln_post.weight.dtype, torch.float32)
        self.assertEqual(model.encoder.conv1.weight.dtype, torch.float16)
        with torch.inference_mode():
            result = model(torch.zeros(1, 80, 4, dtype=torch.float16), torch.tensor([[1, 2]]))
        self.assertEqual(tuple(result.shape), (1, 2, 100))


if __name__ == '__main__': unittest.main()
