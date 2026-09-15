import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.pipeline_service import PipelineService, InvalidAudioError


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.paths = []
        self.service = Mock()
        self.service.health.return_value = dict(status='ok', checkpoint='v03/best',
                                               whisper='ready', phobert='ready', ready=True)
        self.service.analyze.side_effect = self.analyze
        self.client = TestClient(create_app(lambda: self.service))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.service.start.assert_called_once()
        self.service.close.assert_called_once()
        for path in self.paths:
            self.assertFalse(path.exists())
            self.assertFalse(path.parent.exists())

    def analyze(self, path):
        self.paths.append(path)
        self.assertTrue(path.is_file())
        prediction = {'final_label': 'KHONG_CANH_BAO', 'final_confidence': .8,
                      'phobert_result': {'label': 'KHONG_CANH_BAO', 'confidence': .8,
                          'probabilities': {'KHONG_CANH_BAO': .8, 'CAN_XAC_MINH': .1, 'CANH_BAO': .1}},
                      'rule_result': {'needs_review': False, 'findings': []}, 'explanation': 'Mô tả.'}
        from main import classify_units
        detector = Mock()
        detector.predict.return_value = prediction
        units = classify_units({'segments': [{'id': 0, 'text': 'Áo màu xanh. Có hai túi!',
                                             'start': 0, 'end': 4, 'tokens': [123]}]}, detector)
        return {'transcript': 'Áo màu xanh. Có hai túi!', 'classification_units': units,
                'summary': {'whisper_segment_count': 1, 'classification_unit_count': 2,
                            'final_label_counts': {'KHONG_CANH_BAO': 2, 'CAN_XAC_MINH': 0, 'CANH_BAO': 0}}}

    def test_health(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['ready'])
        self.assertEqual(response.json()['checkpoint'], 'v03/best')

    def test_invalid_extension(self):
        self.assertEqual(self.client.post('/analyze-audio', files={'file': ('x.exe', b'x')}).status_code, 400)
        self.service.analyze.assert_not_called()

    def test_empty_and_missing(self):
        self.assertEqual(self.client.post('/analyze-audio', files={'file': ('x.wav', b'')}).status_code, 400)
        self.assertEqual(self.client.post('/analyze-audio').status_code, 400)

    def test_size_limit(self):
        with patch('backend.app.MAX_FILE_BYTES', 2):
            self.assertEqual(self.client.post('/analyze-audio', files={'file': ('x.mp3', b'abc')}).status_code, 413)
        self.service.analyze.assert_not_called()

    def test_schema_and_multiple_requests(self):
        for suffix in ('m4a', 'mp3', 'wav', 'mp4'):
            response = self.client.post('/analyze-audio', files={'file': ('../x.' + suffix, b'audio')})
            self.assertEqual(response.status_code, 200, response.text)
            data = response.json()
            self.assertEqual(data['filename'], 'x.' + suffix)
            self.assertEqual(len(data['classification_units']), 2)
            self.assertTrue(data['classification_units'][0]['timestamp_is_approximate'])
            self.assertNotIn('tokens', response.text)
            self.assertGreaterEqual(data['processing_time_seconds'], 0)

    def test_failure_cleanup_and_redaction(self):
        def fail(path):
            self.paths.append(path)
            raise RuntimeError('SECRET C:/private/model')
        self.service.analyze.side_effect = fail
        with self.assertLogs('backend.app', level='ERROR'):
            response = self.client.post('/analyze-audio', files={'file': ('x.wav', b'bad')})
        self.assertEqual(response.status_code, 500)
        self.assertNotIn('SECRET', response.text)

    def test_corrupt_audio(self):
        self.service.analyze.side_effect = InvalidAudioError('private path')
        self.assertEqual(self.client.post('/analyze-audio', files={'file': ('x.wav', b'bad')}).status_code, 400)

    def test_cors(self):
        for origin, allowed in [('http://localhost:5173', True), ('https://example.com', False)]:
            response = self.client.options('/analyze-audio', headers={
                'Origin': origin, 'Access-Control-Request-Method': 'POST'})
            self.assertEqual('access-control-allow-origin' in response.headers, allowed)


class ServiceTests(unittest.TestCase):
    def test_models_loaded_once_for_multiple_calls(self):
        with patch('backend.pipeline_service.check_environment', return_value={'cuda_available': False}), \
             patch('backend.pipeline_service.verify_checkpoint'), \
             patch('backend.pipeline_service.create_detector') as detector, \
             patch('speech.transcribe.WhisperTranscriber') as whisper:
            whisper.return_value.transcribe.return_value = {'text': '', 'segments': []}
            service = PipelineService()
            service.start()
            service.analyze(Path('a.wav'))
            service.analyze(Path('b.wav'))
            whisper.assert_called_once()
            detector.assert_called_once()
            self.assertEqual(whisper.return_value.transcribe.call_count, 2)
            service.close()
            self.assertFalse(service.health()['ready'])
