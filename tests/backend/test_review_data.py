import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from backend.app import create_app
from tests.backend.test_history import prediction
from backend.database import ReviewSample, ReviewRevision


class ReviewTests(unittest.TestCase):
    def test_accumulation_tracks_exact_revision_not_export_or_validation(self):
        from sqlalchemy import select
        from backend.database import SupplementTrainingRun, SupplementTrainingUsage
        _, sample = self.feedback_sample()
        self.assertEqual(self.client.get('/supplementary/samples').json()['accumulation']['unused_training_count'],0)
        sample = self.approve(sample).json()
        self.client.post('/supplementary/exports')
        stats = self.client.get('/supplementary/samples').json()['accumulation']
        self.assertEqual(stats['unused_training_count'],1)
        self.assertEqual(stats['exported_current_revision_count'],1)
        self.assertEqual(stats['sources'],{'REAL_CONFIRMED':1})
        database = self.app.state.database
        with database.sessions.begin() as db:
            revision = db.scalar(select(ReviewRevision).where(ReviewRevision.sample_id==sample['id'],
                ReviewRevision.revision==sample['revision']))
            db.add(SupplementTrainingRun(id='test-run',checkpoint='test/checkpoint',
                config_sha256='a'*64,completed_at='2026-10-06T00:00:00+00:00'))
            db.flush()
            db.add(SupplementTrainingUsage(run_id='test-run',revision_id=revision.id,split='validation'))
        stats=self.client.get('/supplementary/samples').json()['accumulation']
        self.assertEqual(stats['unused_training_count'],1)
        self.assertEqual(stats['validation_only_count'],1)
        with database.sessions.begin() as db:
            usage=db.get(SupplementTrainingUsage,('test-run',revision.id));usage.split='train'
        self.assertEqual(self.client.get('/supplementary/samples').json()['accumulation']['unused_training_count'],0)
        edited=self.client.put('/supplementary/samples/'+sample['id'],json=dict(
            self.content,text='Chị qua kênh bán khác của shop chọn hàng nhé.',
            expected_revision=sample['revision'],editor='Người thử')).json()
        self.approve(edited)
        stats=self.client.get('/supplementary/samples').json()['accumulation']
        self.assertEqual(stats['unused_training_count'],1)
        self.assertEqual(stats['exported_current_revision_count'],0)
        self.assertEqual(stats['family_count'],1)
        self.assertEqual(stats['labels'],{'CANH_BAO':1})
        database.migrate()
        self.assertEqual(self.client.get('/supplementary/samples').json()['accumulation'],stats)

    def database_url(self):
        return 'sqlite:///' + (Path(self.directory.name)/'test.sqlite3').as_posix()

    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.export_patch=patch('backend.review_data.EXPORT_ROOT',Path(self.directory.name)/'exports')
        self.export_patch.start()
        self.service=Mock()
        self.service.health.return_value=dict(ready=True,model_version='binary_v04',checkpoint='models/actual/best',classification_mode='binary')
        self.service.analyze.return_value=prediction('KHONG_CANH_BAO')
        self.url=self.database_url()
        self.app=create_app(lambda:self.service,self.url)
        self.client=TestClient(self.app);self.client.__enter__()
        self.upload=self.client.post('/analyze-audio',files={'file':('sample.wav',b'audio')}).json()
        self.segment=self.upload['classification_units'][0]['segment_id']
        self.content=dict(text='Nếu giỏ hết hàng, qua gian hàng khác của shop đặt nhé.',label='CANH_BAO',
            family_id='F_TEST',situation='DIEU_HUONG_NGOAI_TIKTOK',reason='Mời mua bên ngoài.')

    def tearDown(self):
        self.client.__exit__(None,None,None);self.export_patch.stop();self.directory.cleanup()

    def feedback_sample(self,label='CANH_BAO'):
        payload=dict(label=label,note='Người dùng sửa nhãn.',corrected_text=self.content['text'],situation=self.content['situation'])
        f=self.client.post(f'/segments/{self.segment}/feedback',json=payload).json()
        response=self.client.post('/supplementary/from-feedback/'+f['id'])
        self.assertEqual(response.status_code,200,response.text)
        return f,response.json()

    def approve(self,sample):
        return self.client.post('/supplementary/samples/'+sample['id']+'/approve',json=dict(
            expected_revision=sample['revision'],reviewer='Người duyệt thử',confirm_text=True))

    def test_correct_text_label_approve_export_and_originals_unchanged(self):
        f,s=self.feedback_sample()
        self.assertEqual(s['status'],'CHO_DUYET');self.assertIsNone(s['reviewer'])
        self.assertEqual(s['original_text'],'Nội dung thử.')
        self.assertEqual(s['predicted_label'],'KHONG_CANH_BAO')
        self.assertEqual(self.client.post('/supplementary/exports').status_code,400)
        approved=self.approve(s).json();self.assertEqual(approved['source_type'],'USER_CONFIRMED_SPEECH')
        exported=self.client.post('/supplementary/exports').json()
        download=self.client.get('/supplementary/exports/'+exported['id']+'/download')
        rows=list(csv.DictReader(io.StringIO(download.content.decode('utf-8'))))
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['text'],self.content['text'])
        self.assertEqual(rows[0]['label'],'CANH_BAO');self.assertEqual(rows[0]['review_status'],'DA_DUYET')
        self.assertEqual(rows[0]['feedback_id'],f['id'])
        import hashlib
        self.assertEqual(hashlib.sha256(download.content).hexdigest(),exported['sha256'])
        original=self.client.get('/sessions/'+self.upload['session_id']).json()['classification_units'][0]
        self.assertEqual(original['text'],'Nội dung thử.')
        self.assertEqual(original['phobert_result'],self.upload['classification_units'][0]['phobert_result'])

    def test_pending_excluded_and_confirm_correct_prediction_eligible(self):
        _,s=self.feedback_sample('KHONG_CANH_BAO');self.approve(s)
        self.client.post('/supplementary/samples',json=dict(self.content,text='Một đề xuất khác.',source_type='AI_GENERATED'))
        e=self.client.post('/supplementary/exports').json();self.assertEqual(e['sample_count'],1)
        rows=self.client.get('/supplementary/samples').json()
        self.assertEqual(rows['counts'],dict(DA_DUYET=1,CHO_DUYET=1))
        self.assertEqual(self.client.get('/supplementary/samples?status=CHO_DUYET&label=CANH_BAO&situation=DIEU_HUONG_NGOAI_TIKTOK').json()['total'],1)
        self.assertEqual(len(self.client.get('/supplementary/samples?page_size=1').json()['items']),1)

    def test_repeated_clicks_and_normalized_conflicts(self):
        f,s=self.feedback_sample()
        again=self.client.post('/supplementary/from-feedback/'+f['id']).json()
        self.assertEqual(again['id'],s['id'])
        approved=self.approve(s).json();self.assertEqual(self.approve(s).json()['revision'],approved['revision'])
        for label in ('CANH_BAO','KHONG_CANH_BAO'):
            duplicate=dict(self.content,text='  '+self.content['text'].upper()+'  ',label=label)
            r=self.client.post('/supplementary/samples',json=duplicate)
            self.assertEqual(r.status_code,409);self.assertIn(s['id'],r.json()['detail'])
        self.assertEqual(self.client.get('/supplementary/samples').json()['total'],1)
        first=self.client.post('/supplementary/exports').json()
        self.assertEqual(first['id'],self.client.post('/supplementary/exports').json()['id'])

    def test_edit_requires_reapproval_history_and_old_export_remains(self):
        _,s=self.feedback_sample();s=self.approve(s).json()
        exported=self.client.post('/supplementary/exports').json()
        old_csv=self.client.get('/supplementary/exports/'+exported['id']+'/download').content
        change=dict(self.content,text='Shop cũng có gian hàng ngoài TikTok.',family_id=s['family_id'],expected_revision=s['revision'],editor='Người sửa thử')
        r=self.client.put('/supplementary/samples/'+s['id'],json=change)
        self.assertEqual(r.status_code,200,r.text);new=r.json()
        self.assertEqual(new['status'],'CHO_DUYET');self.assertIsNone(new['reviewer']);self.assertIsNone(new['reviewed_at'])
        self.assertEqual(self.client.post('/supplementary/exports').status_code,400)
        self.assertEqual(self.approve(s).status_code,409)
        self.approve(new)
        d=self.client.get('/supplementary/samples/'+s['id']).json()
        self.assertEqual([h['action'] for h in d['history']],['CREATE_FROM_FEEDBACK','APPROVE','EDIT_REQUIRES_APPROVAL','APPROVE'])
        self.assertEqual(self.client.get('/supplementary/exports/'+exported['id']+'/download').content,old_csv)

    def test_validation_bad_ids_family_and_no_auto_human_approval(self):
        for url in ('/supplementary/samples/missing','/supplementary/exports/missing/download'):
            self.assertEqual(self.client.get(url).status_code,404)
        self.assertEqual(self.client.post('/supplementary/from-feedback/missing').status_code,404)
        for values in (dict(self.content,label='CAN_XAC_MINH'),dict(self.content,text='   ')):
            self.assertEqual(self.client.post('/supplementary/samples',json=values).status_code,422)
        _,s=self.feedback_sample()
        for values in (dict(expected_revision=1,reviewer=' ',confirm_text=True),dict(expected_revision=1,reviewer='x',confirm_text=False)):
            self.assertEqual(self.client.post('/supplementary/samples/'+s['id']+'/approve',json=values).status_code,422)
        proposal=dict(self.content,text='Câu AI đối chiếu.',source_type='AI_GENERATED',source_sample_id=s['id'])
        self.assertEqual(self.client.post('/supplementary/samples',json=proposal).status_code,409)
        proposal['family_id']=s['family_id']
        r=self.client.post('/supplementary/samples',json=proposal).json()
        self.assertEqual(r['source_type'],'AI_GENERATED');self.assertIsNone(r['reviewer'])

    def test_restart_persists_samples_exports_and_migration_repeatable(self):
        _,s=self.feedback_sample();self.approve(s)
        e=self.client.post('/supplementary/exports').json()
        self.app.state.database.migrate()
        self.client.__exit__(None,None,None)
        self.app=create_app(lambda:self.service,self.url);self.client=TestClient(self.app);self.client.__enter__()
        self.assertEqual(self.client.get('/supplementary/samples/'+s['id']).json()['status'],'DA_DUYET')
        self.assertEqual(self.client.get('/supplementary/exports').json()[0]['sha256'],e['sha256'])
        self.assertEqual(self.client.get('/supplementary/exports/'+e['id']+'/download').status_code,200)

    def test_revision_failure_rolls_back_edit_and_approval(self):
        _,s=self.feedback_sample()
        with patch('backend.review_data.audit',side_effect=RuntimeError('fail')):
            with self.assertRaises(RuntimeError):self.approve(s)
        current=self.client.get('/supplementary/samples/'+s['id']).json()
        self.assertEqual(current['status'],'CHO_DUYET');self.assertEqual(current['revision'],1)


if __name__=='__main__':unittest.main()
