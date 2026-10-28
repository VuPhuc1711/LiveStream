"""API/persistence integration with MOCK ASR/VAD, not recognition-quality proof.

Every test creates and migrates its own temporary SQLite database. No production
database, real model, training examples or saved debug recording is modified.
"""
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.pipeline_service import PipelineService
from tests.backend.test_live import fake_vad, packet, result


RAW_TEXT = 'Chuỗi giải mã giả lập vượt thời lượng audio.'


@contextmanager
def mock_decoded_pcm(_path):
    # Simulated speech admission only, never evidence that speech was recognized.
    yield iter([packet(0, 1)[4:]])


class AsrQualityFlowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database_url = 'sqlite:///' + (Path(self.directory.name) / 'quality.sqlite3').as_posix()
        self.service = PipelineService()
        self.service.start = Mock()
        self.service.close = Mock()
        self.service.model_version = 'binary_v04'
        self.service.checkpoint = 'mock_checkpoint_not_loaded/binary_v04'
        self.service.classification_mode = 'binary'
        self.service.labels = ('KHONG_CANH_BAO', 'CANH_BAO')
        self.service.live_vad = fake_vad()
        self.service.live_vad.configuration.return_value = {'engine': 'MOCK_VAD'}
        self.service.transcriber = Mock()
        self.service.detector = Mock()
        self.service.detector.predict.side_effect = lambda _text: {
            k: v for k, v in result()['units'][0].items()
            if k in ('final_label', 'final_confidence', 'phobert_result', 'explanation')}
        self.invalid_decode()
        self.client = None
        self.restart_client()
        self.addCleanup(self.close_client)

    def close_client(self):
        if self.client is not None:
            self.client.__exit__(None, None, None)
            self.client = None

    def restart_client(self):
        self.close_client()
        self.client = TestClient(create_app(lambda: self.service, self.database_url))
        self.client.__enter__()

    def invalid_decode(self):
        self.service.transcriber.transcribe.return_value = {
            'text': RAW_TEXT,
            'segments': [{'id': 0, 'text': RAW_TEXT, 'start': 0, 'end': 30,
                          'no_speech_prob': .01, 'tokens': [123]}]}
        self.service.transcriber.transcribe_without_prompt.return_value = {'text': '', 'segments': []}

    def websocket_session(self, source='browser_tab_live'):
        events = []
        with self.client.websocket_connect('/ws/monitor', headers={'origin': 'http://127.0.0.1:5173'}) as ws:
            ws.send_json(dict(type='start', format='pcm_s16le', sample_rate=16000,
                              channels=1, source=source))
            started = ws.receive_json()
            self.assertEqual(started['type'], 'started')
            sid = started['session_id']
            ws.send_bytes(packet(0))
            while True:
                item = ws.receive_json()
                events.append(item)
                self.assertNotEqual(item['type'], 'error', item)
                if item['type'] == 'ack':
                    break
            ws.send_json(dict(type='stop', last_sequence=0))
            while True:
                item = ws.receive_json()
                events.append(item)
                self.assertNotEqual(item['type'], 'error', item)
                if item['type'] == 'completed':
                    break
        self.assertTrue(all(e['session_id'] == sid for e in events))
        return sid, events

    def assert_unresolved_detail(self, sid):
        response = self.client.get('/sessions/' + sid)
        self.assertEqual(response.status_code, 200)
        detail = response.json()
        self.assertEqual(detail['status'], 'completed')
        self.assertEqual(detail['classification_units'], [])
        self.assertEqual(detail['transcript'], '')
        self.assertEqual(detail['summary']['final_label_counts'], {'KHONG_CANH_BAO': 0, 'CANH_BAO': 0})
        self.assertEqual(detail['asr_unresolved_windows'], 1)
        self.assertEqual(detail['asr_recovered_windows'], 0)
        self.assertEqual(len(detail['asr_quality_warnings']), 1)
        self.assertTrue(detail['asr_quality_warnings'][0]['retried'])
        trace = detail['asr_windows'][0]
        self.assertEqual(trace['whisper_raw']['text'], RAW_TEXT)
        self.assertEqual(trace['whisper_raw']['segments'][0]['end'], 30)
        self.assertNotIn('tokens', trace['whisper_raw']['segments'][0])
        self.assertTrue(trace['asr_rejected_segments'])
        self.assertEqual(trace['unit_ids'], [])
        self.assertIsNone(trace['asr_used_attempt'])
        exported = self.client.get('/sessions/' + sid + '/export')
        self.assertEqual(exported.status_code, 200)
        self.assertEqual(exported.json(), detail)
        return detail

    def test_realtime_unresolved_event_counts_and_history_survive_restart(self):
        sid, events = self.websocket_session()
        batch = next(e for e in events if e['type'] == 'result')
        self.assertEqual(batch['classification_units'], [])
        self.assertTrue(batch['asr_quality_warning'])
        self.assertTrue(batch['asr_quality_warning']['retried'])
        self.assertEqual(batch['asr_unresolved_windows'], 1)
        self.assertEqual(batch['asr_recovered_windows'], 0)
        final = events[-1]
        self.assertEqual(final['asr_unresolved_windows'], 1)
        self.assertEqual(final['warning_count'], 0)
        self.assertEqual(final['unit_count'], 0)
        self.assertEqual(final['backlog_seconds'], 0)
        self.assertEqual(final['pending_windows'], 0)
        self.assertEqual(final['unprocessed_audio_seconds'], 0)
        self.assertFalse(final['processing'])
        self.assertAlmostEqual(final['received_seconds'],
                               final['processed_audio_seconds'] + final['silence_seconds'])
        self.service.detector.predict.assert_not_called()
        self.service.transcriber.transcribe_without_prompt.assert_called_once()
        self.service.transcriber.transcribe_retry.assert_not_called()
        before = self.assert_unresolved_detail(sid)
        self.restart_client()
        self.assertEqual(self.assert_unresolved_detail(sid), before)

    def test_consecutive_session_does_not_inherit_unresolved_notice_or_rejected_text(self):
        first, _ = self.websocket_session()
        self.service.detector.predict.assert_not_called()
        self.service.transcriber.transcribe.return_value = {
            'text': 'Không.', 'segments': [{'id': 0, 'text': 'Không.', 'start': 0, 'end': .2,
                                            'no_speech_prob': .99}]}
        second, events = self.websocket_session('microphone_live')
        self.assertNotEqual(first, second)
        self.assertEqual(events[-1]['asr_unresolved_windows'], 0)
        self.assertEqual(events[-1]['asr_recovered_windows'], 0)
        self.assertTrue(all(not e.get('asr_quality_warning') for e in events))
        self.service.detector.predict.assert_called_once_with('Không.')
        self.service.transcriber.transcribe_without_prompt.assert_called_once()
        detail = self.client.get('/sessions/' + second).json()
        self.assertEqual(detail['asr_quality_warnings'], [])
        self.assertEqual(detail['asr_unresolved_windows'], 0)
        self.assertEqual(detail['transcript'], 'Không.')
        self.assertNotIn(RAW_TEXT, detail['transcript'])
        self.assertEqual(self.client.get('/sessions/' + first).json()['asr_unresolved_windows'], 1)

    def test_upload_response_history_and_export_keep_unresolved_notice_and_raw_trace(self):
        self.service.transcriber.transcribe_without_prompt.return_value = self.service.transcriber.transcribe.return_value
        with patch('backend.upload_audio.decoded_pcm', side_effect=mock_decoded_pcm):
            response = self.client.post('/analyze-audio', files={
                'file': ('mock-audio.wav', b'MOCK_AUDIO_NOT_A_REAL_RECORDING', 'audio/wav')})
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload['transcript'], '')
        self.assertEqual(payload['classification_units'], [])
        self.assertEqual(payload['summary']['classification_unit_count'], 0)
        self.assertEqual(payload['asr_unresolved_windows'], 1)
        self.assertEqual(payload['asr_recovered_windows'], 0)
        self.assertTrue(payload['asr_quality_warnings'][0]['retried'])
        self.assertEqual(payload['asr_quality_warnings'][0]['reason'], 'retry_timestamps_invalid')
        self.assertEqual(payload['audio_accounting']['unprocessed_audio_seconds'], 0)
        self.service.detector.predict.assert_not_called()
        self.service.transcriber.transcribe_without_prompt.assert_called_once()
        self.service.transcriber.transcribe_retry.assert_not_called()
        detail = self.assert_unresolved_detail(payload['session_id'])
        self.assertEqual(detail['asr_quality_warnings'], payload['asr_quality_warnings'])
        self.assertEqual(detail['asr_windows'][0]['window_id'], payload['asr_quality_warnings'][0]['window_id'])

    def test_no_prompt_recovery_classifies_new_native_decode_and_keeps_both_attempts(self):
        replacement = 'Không chuyển đơn ra ngoài.'
        self.service.transcriber.transcribe_without_prompt.return_value = {
            'text': replacement,
            'segments': [{'id': 0, 'text': replacement, 'start': 0, 'end': .4, 'tokens': [456]}]}
        sid, events = self.websocket_session()
        final = events[-1]
        self.assertEqual(final['asr_unresolved_windows'], 0)
        self.assertEqual(final['asr_recovered_windows'], 1)
        self.assertEqual(final['unit_count'], 1)
        self.assertEqual(final['backlog_seconds'], 0)
        self.assertEqual(final['pending_windows'], 0)
        batch = next(e for e in events if e['type'] == 'result')
        self.assertIsNone(batch['asr_quality_warning'])
        self.assertEqual(batch['classification_units'][0]['text'], replacement)
        self.service.detector.predict.assert_called_once_with(replacement)
        self.service.transcriber.transcribe_without_prompt.assert_called_once_with(
            self.service.transcriber.transcribe.call_args.args[0])
        self.service.transcriber.transcribe_retry.assert_not_called()
        detail = self.client.get('/sessions/' + sid).json()
        self.assertEqual(detail['transcript'], replacement)
        self.assertEqual(detail['asr_quality_warnings'], [])
        self.assertEqual(detail['asr_unresolved_windows'], 0)
        self.assertEqual(detail['asr_recovered_windows'], 1)
        trace = detail['asr_windows'][0]
        self.assertEqual(trace['whisper_raw']['text'], RAW_TEXT)
        self.assertEqual(trace['whisper_raw']['segments'][0]['end'], 30)
        self.assertEqual(trace['asr_recovery']['whisper_raw']['text'], replacement)
        self.assertEqual(trace['asr_recovery']['whisper_raw']['segments'][0]['end'], .4)
        self.assertNotIn('tokens', trace['asr_recovery']['whisper_raw']['segments'][0])
        self.assertEqual(trace['asr_used_attempt'], 'without_prompt')
        self.assertEqual(self.client.get('/sessions/' + sid + '/export').json(), detail)
        self.restart_client()
        self.assertEqual(self.client.get('/sessions/' + sid).json(), detail)

    def test_unchanged_retry_text_with_new_timestamps_cannot_become_a_prediction(self):
        self.service.transcriber.transcribe_without_prompt.return_value = {
            'text': RAW_TEXT,
            'segments': [{'id': 0, 'text': RAW_TEXT, 'start': 0, 'end': .4}]}
        sid, events = self.websocket_session()
        self.assertEqual(events[-1]['asr_unresolved_windows'], 1)
        self.assertEqual(events[-1]['asr_recovered_windows'], 0)
        self.service.detector.predict.assert_not_called()
        self.service.transcriber.transcribe_without_prompt.assert_called_once()
        detail = self.assert_unresolved_detail(sid)
        self.assertEqual(detail['asr_quality_warnings'][0]['reason'], 'retry_unchanged_text')


if __name__ == '__main__':
    unittest.main()
