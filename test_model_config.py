"""Configuration/manifest tests: fixtures only, no model inference."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import model_config
import main


class ModelConfigTests(unittest.TestCase):
    def test_shared_default_and_explicit_override(self):
        from nlp.phobert_detector import DEFAULT_CHECKPOINT
        self.assertEqual(main.CHECKPOINT, model_config.get_default_checkpoint())
        self.assertEqual(DEFAULT_CHECKPOINT, main.CHECKPOINT)
        custom = Path('models/explicit-test/best').resolve()
        with patch('nlp.phobert_detector.PhoBertDetector') as cls, patch('nlp.hybrid_detector.HybridDetector') as hybrid:
            cls.return_value.checkpoint_path = custom
            hybrid.return_value.phobert_detector = cls.return_value
            detector = main.create_detector(custom, 'cpu')
            cls.assert_called_once_with(checkpoint_path=custom, device='cpu')
            self.assertIs(detector, hybrid.return_value)

    def test_default_constructor_resolves_config_and_explicit_path_wins(self):
        from nlp.phobert_detector import PhoBertDetector
        with tempfile.TemporaryDirectory() as tmp:
            default = Path(tmp) / 'configured'
            custom = Path(tmp) / 'explicit'
            with patch('nlp.phobert_detector.get_default_checkpoint', return_value=default) as get:
                with self.assertRaises(FileNotFoundError): PhoBertDetector()
                get.assert_called_once()
                get.reset_mock()
                with self.assertRaises(FileNotFoundError): PhoBertDetector(custom)
                get.assert_not_called()

    def test_manifest_v02_v03_and_tampered_weights(self):
        for schema in ['artifact_sha256', 'file_sha256']:
            with self.subTest(schema=schema), tempfile.TemporaryDirectory() as tmp:
                checkpoint = Path(tmp) / 'best'; checkpoint.mkdir()
                weight = checkpoint / 'model.safetensors'; weight.write_bytes(b'fixture')
                key = 'best/model.safetensors' if schema == 'artifact_sha256' else 'model.safetensors'
                lock = {'status':'FROZEN','checkpoint':str(checkpoint),schema:{key:hashlib.sha256(b'fixture').hexdigest()}}
                (checkpoint.parent / 'checkpoint_lock.json').write_text(json.dumps(lock),encoding='utf-8')
                self.assertEqual(model_config.verify_model_checkpoint(checkpoint)['path'],str(checkpoint))
                weight.write_bytes(b'tampered')
                with self.assertRaises(ValueError): model_config.verify_model_checkpoint(checkpoint)

    def test_reviewed_audio_allows_metadata_and_unbalanced_labels(self):
        import csv
        from evaluate_audio_pipeline import load_ground_truth
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'truth.csv'
            labels = ['KHONG_CANH_BAO']*4 + ['CAN_XAC_MINH']*2 + ['CANH_BAO']*3
            with path.open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['id','text','true_label','reason','review_status','review_note'])
                writer.writeheader()
                writer.writerows({'id':str(i),'text':f'Câu giả {i}.','true_label':label,'reason':'Fixture',
                                  'review_status':'DA_DUYET','review_note':''} for i,label in enumerate(labels))
            self.assertEqual(len(load_ground_truth(path)),9)


if __name__=='__main__':unittest.main()
