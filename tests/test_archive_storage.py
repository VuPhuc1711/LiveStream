"""Mixed external/local archive integrity without touching real archived assets."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from tools import verify_archive

class ArchiveStorageTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        base=Path(self.directory.name)
        self.root=base/'project';self.storage=base/'external'
        self.root.mkdir();self.storage.mkdir()
        (self.root/'config').mkdir()
        self.entries=[]
        for relative,local in [('archive/research/a.py',False),('config/training/old.json',False),('backups/config/private.env',True)]:
            path=(self.root if local else self.storage)/relative;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(b'fixture historical bytes')
            self.entries.append({'stored':relative,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        self.manifest=self.storage/'archive/manifest_20261024.json';self.write_manifest()
        self.transfer=self.storage/'manifest_20261028.json'
        self.transfer.write_text(json.dumps({'completed':True,'files':[
            {'relative':entry['stored'],'sha256':entry['sha256']} for entry in self.entries[:2]]}),encoding='utf-8')
        self.pointer={'external_root':str(self.storage),'legacy_manifest':'archive/manifest_20261024.json',
                      'legacy_local_roots':['backups'],'transfer_manifest':str(self.transfer)}
        self.write_pointer()
        self.root_patch=patch.object(verify_archive,'ROOT',self.root);self.root_patch.start();self.addCleanup(self.root_patch.stop)

    def write_pointer(self):
        (self.root/'config/storage.json').write_text(json.dumps(self.pointer),encoding='utf-8')

    def write_manifest(self):
        self.manifest.write_text(json.dumps({'entries':self.entries}),encoding='utf-8')

    def test_originals_can_span_external_storage_and_kept_local_backup(self):
        result=verify_archive.verify(include_transferred=True)
        self.assertEqual(result['files'],3)
        self.assertEqual(result['transferred_files_checked'],2)
        self.assertTrue(result['all_preserved'])

    def test_tampered_external_asset_is_reported(self):
        (self.storage/'archive/research/a.py').write_bytes(b'changed')
        result=verify_archive.verify()
        self.assertFalse(result['all_preserved'])
        self.assertEqual(result['failures'],['archive/research/a.py'])

    def test_path_escape_is_rejected(self):
        self.entries[0]['stored']='../escape';self.write_manifest()
        with self.assertRaisesRegex(ValueError,'escapes'):verify_archive.verify()

    def test_archive_inside_project_cannot_be_misrepresented_as_external(self):
        self.pointer['external_root']=str(self.root/'nested');self.write_pointer()
        with self.assertRaisesRegex(ValueError,'outside'):verify_archive.locations()

    def test_additional_move_manifest_preserves_and_detects_changed_files(self):
        path=self.storage/'archive/legacy_cli/main.py'
        path.parent.mkdir(parents=True);path.write_bytes(b'original legacy CLI')
        supplement=self.storage/'manifest_legacy_cli.json'
        supplement.write_text(json.dumps({'completed':True,'files':[
            {'relative':'archive/legacy_cli/main.py','sha256':hashlib.sha256(path.read_bytes()).hexdigest()}]}),encoding='utf-8')
        self.pointer['additional_manifests']=['manifest_legacy_cli.json'];self.write_pointer()
        result=verify_archive.verify()
        self.assertTrue(result['all_preserved'])
        self.assertEqual(result['additional_files_checked'],1)
        path.write_bytes(b'tampered')
        self.assertEqual(verify_archive.verify()['failures'],['archive/legacy_cli/main.py'])

    def test_additional_manifest_cannot_escape_external_storage(self):
        self.pointer['additional_manifests']=['../escape.json'];self.write_pointer()
        with self.assertRaisesRegex(ValueError,'escapes'):verify_archive.verify()
