"""Prepared train/test data integrity; no model, benchmark or source-path dependency."""
import csv
import hashlib
import json
import unittest
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def normalized(text):
    return ' '.join(unicodedata.normalize('NFC', text).split()).casefold()


def canonical(rows):
    return json.dumps(rows, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':')).encode('utf-8')


class CurrentDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT/'config/data/current_dataset.json').read_text(encoding='utf-8'))
        cls.rows = {}
        for split, metadata in cls.manifest['files'].items():
            with (ROOT/metadata['path']).open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != cls.manifest['current_columns']:
                    raise AssertionError('CSV schema does not match prepared manifest')
                cls.rows[split] = list(reader)

    def test_two_splits_counts_hashes_labels_and_approval(self):
        self.assertEqual(set(self.rows), {'train', 'test'})
        self.assertEqual({path.name for path in (ROOT/'data/current').glob('*.csv')}, {'train.csv', 'test.csv'})
        self.assertEqual({split: len(rows) for split, rows in self.rows.items()}, {'train': 584, 'test': 182})
        expected = {'train': {'KHONG_CANH_BAO': 420, 'CANH_BAO': 164},
                    'test': {'KHONG_CANH_BAO': 134, 'CANH_BAO': 48}}
        for split, rows in self.rows.items():
            metadata = self.manifest['files'][split]
            self.assertEqual(hashlib.sha256((ROOT/metadata['path']).read_bytes()).hexdigest(), metadata['sha256'])
            self.assertEqual(dict(Counter(row['label'] for row in rows)), expected[split])
            self.assertTrue(all(row['split'] == split for row in rows))
            self.assertTrue(all(row['review_status'] in {'DA_DUYET', 'DA_DUYET_AI'} for row in rows))

    def test_all_original_columns_and_rows_are_preserved(self):
        # Reconstruct original rows from current CSV and compare the recorded
        # source fingerprint, even after the original files move off-project.
        for split, rows in self.rows.items():
            metadata = self.manifest['files'][split]
            original_split = metadata['original_split']
            original = []
            for row in rows:
                self.assertEqual(row['original_split'], original_split)
                source_row = {column: row[column] for column in self.manifest['original_columns']}
                source_row['split'] = original_split
                original.append(source_row)
            self.assertEqual(hashlib.sha256(canonical(original)).hexdigest(), metadata['source_rows_sha256'])
        self.assertFalse(self.manifest['approved_labels_changed'])
        self.assertFalse(self.manifest['split_membership_changed'])

    def test_unique_ids_text_and_family_separation(self):
        all_rows = self.rows['train'] + self.rows['test']
        self.assertEqual(len(all_rows), 766)
        self.assertEqual(len({row['id'] for row in all_rows}), 766)
        self.assertEqual(len({normalized(row['text']) for row in all_rows}), 766)
        for key in ('id', 'family_id', 'split_group_id'):
            self.assertFalse({row[key] for row in self.rows['train']} &
                             {row[key] for row in self.rows['test']}, key)
        self.assertFalse({normalized(row['text']) for row in self.rows['train']} &
                         {normalized(row['text']) for row in self.rows['test']})

    def test_previous_validation_is_not_represented_as_independent_test(self):
        self.assertFalse(self.manifest['test_evaluation']['independent'])
        self.assertEqual(self.manifest['test_evaluation']['role'], 'REGRESSION_PREVIOUS_VALIDATION')
        self.assertFalse(self.manifest['training_authorized'])
        self.assertFalse(self.manifest['checkpoint_changed'])
        self.assertFalse(self.manifest['model_note']['existing_checkpoint_was_trained_using_this_new_view'])
        self.assertTrue(all(row['original_split'] == 'validation' and
                            row['evaluation_role'] == 'REGRESSION_PREVIOUS_VALIDATION' and
                            row['seen_by_development'] == 'true' for row in self.rows['test']))


if __name__ == '__main__':
    unittest.main()
