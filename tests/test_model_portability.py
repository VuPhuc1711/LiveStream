"""Portable frozen checkpoint verification with tiny files; no model loading."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backend import model_config


class ModelPortabilityTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / 'different-clone'
        self.relative = 'models/phobert_binary_v04_fixture/seed42/best'
        self.checkpoint = self.root / self.relative
        self.checkpoint.mkdir(parents=True)
        contents = {'config.json': b'{"num_labels":2}',
                    'nested/model-00001.safetensors': b'fixture weights'}
        self.hashes = {}
        for name, value in contents.items():
            path = self.checkpoint / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(value)
            self.hashes[name] = hashlib.sha256(value).hexdigest()
        self.lock = {'status': 'FROZEN',
                     'checkpoint': 'D:\\original-project\\' + self.relative.replace('/', '\\'),
                     'file_sha256': dict(self.hashes)}
        self.lock_path = self.checkpoint.parent / 'checkpoint_lock.json'
        self.write_lock()
        self.config = self.root / 'pipeline_model.json'
        self.config.write_text(json.dumps({'version': 'binary_v04', 'checkpoint': self.relative}), encoding='utf-8')
        for attribute, value in [('ROOT', self.root), ('CONFIG_PATH', self.config)]:
            context = patch.object(model_config, attribute, value)
            context.start()
            self.addCleanup(context.stop)

    def write_lock(self):
        self.lock_path.write_text(json.dumps(self.lock), encoding='utf-8')

    def test_windows_frozen_path_relocates_without_modifying_lock(self):
        before = self.lock_path.read_bytes()
        result = model_config.verify_model_checkpoint()
        self.assertEqual(result['file_sha256'], self.hashes)
        self.assertEqual(Path(result['path']), self.checkpoint.resolve())
        self.assertEqual(self.lock_path.read_bytes(), before)

    def test_posix_frozen_path_and_legacy_windows_artifact_names(self):
        self.lock['checkpoint'] = '/old/container/app/' + self.relative
        del self.lock['file_sha256']
        self.lock['artifact_sha256'] = {'best\\' + name.replace('/', '\\'): digest
                                        for name, digest in self.hashes.items()}
        self.lock['artifact_sha256']['history.json'] = '0' * 64
        self.write_lock()
        self.assertEqual(model_config.verify_model_checkpoint()['file_sha256'], self.hashes)

    def test_wrong_run_identity_is_rejected_even_with_identical_artifacts(self):
        self.lock['checkpoint'] = self.lock['checkpoint'].replace('v04_fixture', 'v03_other')
        self.write_lock()
        with self.assertRaisesRegex(ValueError, 'không trỏ'):
            model_config.verify_model_checkpoint()

    def test_checkpoint_outside_repository_is_rejected(self):
        with patch.object(model_config, 'ROOT', self.root / 'other'):
            with self.assertRaisesRegex(ValueError, 'repository'):
                model_config.verify_model_checkpoint(self.checkpoint)

    def test_changed_missing_and_extra_artifacts_are_rejected(self):
        path = self.checkpoint / 'config.json'
        original = path.read_bytes()
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            model_config.verify_model_checkpoint()
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            model_config.verify_model_checkpoint()
        path.write_bytes(original)
        (self.checkpoint / 'unexpected.json').write_bytes(b'extra')
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            model_config.verify_model_checkpoint()

    def test_unfrozen_record_is_rejected(self):
        self.lock['status'] = 'DRAFT'
        self.write_lock()
        with self.assertRaisesRegex(ValueError, 'không trỏ'):
            model_config.verify_model_checkpoint()

    def test_path_traversal_and_ambiguous_model_identity_are_rejected(self):
        for value in ['D:/old/models/../' + self.relative,
                      'D:/models/old/' + self.relative]:
            with self.subTest(value=value):
                self.lock['checkpoint'] = value
                self.write_lock()
                with self.assertRaisesRegex(ValueError, 'định danh'):
                    model_config.verify_model_checkpoint()

    def test_artifact_traversal_and_duplicate_normalized_keys_are_rejected(self):
        self.lock['file_sha256']['../outside'] = '0' * 64
        self.write_lock()
        with self.assertRaisesRegex(ValueError, 'tương đối'):
            model_config.verify_model_checkpoint()
        self.lock['file_sha256'] = dict(self.hashes)
        self.lock['file_sha256']['nested\\model-00001.safetensors'] = self.hashes['nested/model-00001.safetensors']
        self.write_lock()
        with self.assertRaisesRegex(ValueError, 'trùng'):
            model_config.verify_model_checkpoint()


if __name__ == '__main__':
    unittest.main()
