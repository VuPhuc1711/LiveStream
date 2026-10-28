import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError, IntegrityError
from backend.app import create_app
from backend import history
from backend.database import AnalysisSegment, Database


def prediction(label='CANH_BAO'):
    p = .9 if label == 'CANH_BAO' else .1
    return {'classification_mode': 'binary', 'transcript': 'Nội dung thử.',
        'summary': {'classification_unit_count': 1, 'whisper_segment_count': 1,
                    'final_label_counts': {'KHONG_CANH_BAO': int(label != 'CANH_BAO'), 'CANH_BAO': int(label == 'CANH_BAO')}},
        'classification_units': [{'unit_id': 'U001', 'source_segment_id': 0, 'text': 'Nội dung thử.',
            'start': 0, 'end': 2, 'timestamp_is_approximate': True, 'final_label': label,
            'final_confidence': .9, 'phobert_result': {'label': label, 'confidence': .9,
                'probabilities': {'KHONG_CANH_BAO': 1-p, 'CANH_BAO': p}}, 'explanation': 'Mô tả kết quả.'}]}


class HistoryTests(unittest.TestCase):
    def database_url(self):
        return 'sqlite:///' + (Path(self.directory.name) / 'history.sqlite3').as_posix()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.url = self.database_url()
        self.service = Mock()
        self.service.health.return_value = {'ready': True, 'model_version': 'binary_v04',
            'checkpoint': 'models/actual/best', 'classification_mode': 'binary'}
        self.service.analyze.return_value = prediction()
        self.app = create_app(lambda: self.service, self.url)
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.directory.cleanup()

    def upload(self, source='upload'):
        return self.client.post('/analyze-audio', files={'file': ('sample.m4a', b'audio')}, data={'source': source})

    def test_upload_trace_persists_with_segments_in_same_transaction(self):
        self.service.analyze.return_value = dict(prediction(), asr_windows=[dict(window_id='upload_window_0',
            start=2.3,end=4.3,sample_rate=16000,asr_called=True,
            whisper_raw={'text':'Nội dung thử.','segments':[]})])
        response = self.upload('recording')
        self.assertEqual(response.status_code,200)
        sid=response.json()['session_id']
        detail=self.client.get('/sessions/'+sid).json()
        self.assertEqual(detail['asr_windows'][0]['window_id'],'upload_window_0')
        self.assertEqual(detail['asr_windows'][0]['session_id'],sid)
        self.assertEqual(detail['live_windows'],[])
        self.assertEqual(len(detail['classification_units']),1)

    def test_conflicting_upload_trace_rolls_back_segments_and_windows(self):
        trace={'window_id':'upload_window_0','whisper_raw':{'text':'Nội dung thử.'}}
        self.service.analyze.return_value = dict(prediction(), asr_windows=[trace,trace])
        with self.assertLogs('backend.app',level='ERROR'):
            self.assertEqual(self.upload().status_code,500)
        sessions=self.client.get('/sessions').json()['items']
        self.assertEqual(sessions[0]['status'],'failed')
        detail=self.client.get('/sessions/'+sessions[0]['id']).json()
        self.assertEqual(detail['classification_units'],[])
        self.assertEqual(detail['asr_windows'],[])

    def test_saved_detail_export_feedback_and_restart(self):
        response = self.upload('recording')
        self.assertEqual(response.status_code, 200, response.text)
        sid = response.json()['session_id']
        segment = response.json()['classification_units'][0]['segment_id']
        for label in ('CANH_BAO', 'KHONG_CANH_BAO'):
            r = self.client.post(f'/segments/{segment}/feedback', json={'label': label, 'note': 'Người dùng rà soát.'})
            self.assertEqual(r.status_code, 201)
        detail = self.client.get('/sessions/' + sid).json()
        self.assertEqual(detail['source'], 'recording')
        self.assertEqual(detail['model_version'], 'binary_v04')
        self.assertEqual(detail['checkpoint'], 'models/actual/best')
        self.assertTrue(detail['ended_at'].endswith('+00:00'))
        unit = detail['classification_units'][0]
        self.assertEqual(unit['final_label'], 'CANH_BAO')
        self.assertEqual(len(unit['feedback']), 2)
        self.assertEqual(unit['phobert_result']['probabilities']['CANH_BAO'], .9)
        export = self.client.get('/sessions/' + sid + '/export')
        self.assertIn('attachment', export.headers['content-disposition'])
        self.assertEqual(export.json(), detail)
        self.client.__exit__(None, None, None)
        self.client = TestClient(create_app(lambda: self.service, self.url))
        self.client.__enter__()
        self.assertEqual(self.client.get('/sessions/' + sid).json(), detail)

    def test_pagination_filters_validation_and_not_found(self):
        self.upload()
        self.service.analyze.return_value = prediction('KHONG_CANH_BAO')
        self.upload()
        self.service.analyze.side_effect = RuntimeError('SECRET private/path')
        with self.assertLogs('backend.app', level='ERROR'):
            self.assertEqual(self.upload().status_code, 500)
        first = self.client.get('/sessions?page_size=1').json()
        second = self.client.get('/sessions?page=2&page_size=1').json()
        self.assertEqual(first['total'], 3)
        self.assertNotEqual(first['items'][0]['id'], second['items'][0]['id'])
        self.assertEqual(self.client.get('/sessions?warnings_only=true').json()['total'], 1)
        failed = self.client.get('/sessions?status=failed').json()['items'][0]
        detail = self.client.get('/sessions/' + failed['id']).json()
        self.assertEqual(detail['classification_units'], [])
        self.assertNotIn('SECRET', detail['error'])
        for path in ('/sessions/missing', '/sessions/missing/export'):
            self.assertEqual(self.client.get(path).status_code, 404)
        self.assertEqual(self.client.post('/segments/missing/feedback', json={'label':'CANH_BAO'}).status_code, 404)
        self.assertEqual(self.client.post('/segments/missing/feedback', json={'label':'CAN_XAC_MINH'}).status_code, 422)
        for query in ('page=0', 'page_size=101', 'status=invalid'):
            self.assertEqual(self.client.get('/sessions?' + query).status_code, 422)
        self.assertEqual(self.upload('invalid').status_code, 422)

    def test_atomic_completion_rolls_back_partial_segments(self):
        result = prediction()
        result['classification_units'].append(dict(result['classification_units'][0], unit_id='U002'))
        result['summary']['classification_unit_count'] = 2
        result['summary']['final_label_counts']['CANH_BAO'] = 2
        self.service.analyze.return_value = result
        calls = []
        def insert(**kwargs):
            calls.append(kwargs)
            if len(calls) == 2:
                raise RuntimeError('partial failure')
            return AnalysisSegment(**kwargs)
        with patch('backend.history.AnalysisSegment', side_effect=insert), self.assertLogs('backend.app', level='ERROR'):
            self.assertEqual(self.upload().status_code, 500)
        self.assertEqual(len(calls), 2)
        # First segment was flushed, but the later failure rolls the whole transaction back.
        sid = self.client.get('/sessions').json()['items'][0]['id']
        detail = self.client.get('/sessions/' + sid).json()
        self.assertEqual(detail['status'], 'failed')
        self.assertEqual(detail['classification_units'], [])

    def test_interrupted_sessions_not_replayed(self):
        sid = history.start_session(self.app.state.database, 'waiting.wav', 'upload', self.service.health())
        self.client.__exit__(None, None, None)
        self.client = TestClient(create_app(lambda: self.service, self.url))
        self.client.__enter__()
        detail = self.client.get('/sessions/' + sid).json()
        self.assertEqual(detail['status'], 'failed')
        self.assertIn('gián đoạn', detail['error'])
        self.service.analyze.assert_not_called()

    def test_database_errors_are_sanitized(self):
        with patch('backend.history.list_sessions', side_effect=OperationalError('SECRET/path', {}, Exception('SECRET'))), self.assertLogs('backend.app', level='ERROR'):
            response = self.client.get('/sessions')
        self.assertEqual(response.status_code, 500)
        self.assertNotIn('SECRET', response.text)

    def test_foreign_keys_and_migration_repeatable(self):
        database = self.app.state.database
        database.migrate()
        with self.assertRaises(IntegrityError):
            with database.sessions.begin() as db:
                from backend.database import Feedback
                db.add(Feedback(segment_id='missing', label='CANH_BAO', note='x'))
