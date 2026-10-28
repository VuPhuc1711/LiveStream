"""Organization tests, without loading model weights or modifying databases."""
import ast
import unittest
from pathlib import Path
from backend import model_config
from tools.verify_archive import verify, locations

class ProjectStructureTests(unittest.TestCase):
    def test_archived_originals_are_intact(self):
        # Historical storage is optional on another machine; application tests
        # must not pretend the original archive was checked when it is absent.
        try:manifest=locations()['manifest']
        except FileNotFoundError:self.skipTest('Historical storage registry unavailable')
        if not manifest.is_file():self.skipTest('External historical archive not connected')
        self.assertTrue(verify()['all_preserved'])

    def test_configuration_root_and_active_checkpoint_remain_the_same(self):
        self.assertEqual(model_config.ROOT,Path(__file__).resolve().parents[1])
        config=model_config.get_active_model()
        self.assertEqual(config['version'],'binary_v04')
        self.assertEqual(config['classification_mode'],'binary')
        self.assertEqual(config['classifier'],'phobert_only')
        self.assertTrue(model_config.get_default_checkpoint().is_dir())

    def test_application_has_no_imports_of_legacy_rules_or_research(self):
        for package in ('backend','speech','nlp'):
            for path in (model_config.ROOT/package).glob('*.py'):
                tree=ast.parse(path.read_text(encoding='utf-8'))
                imports=[node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
                self.assertFalse(set(imports)&{'main','model_config','nlp.hybrid_detector','nlp.keyword_detector'},str(path))
                self.assertFalse(any((name or '').startswith('archive.') for name in imports),str(path))
