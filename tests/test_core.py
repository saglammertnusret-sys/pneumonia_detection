import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from pneumonia.data import scan_dataset, counts
from pneumonia.metrics import summarize, save_evaluation

def make_fake_data(root):
    rng = np.random.default_rng(42)
    for split in ('train', 'val', 'test'):
        for label in ('NORMAL', 'PNEUMONIA'):
            directory = root / split / label
            directory.mkdir(parents=True)
            for index in range(4):
                Image.fromarray(rng.integers(0, 256, (48, 64), dtype=np.uint8)).save(directory / f'{index}.png')

class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'dataset'
        make_fake_data(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_class_mapping_and_counts(self):
        result = scan_dataset(self.root)
        self.assertEqual(counts(result)['train'], {'NORMAL': 4, 'PNEUMONIA': 4})
        self.assertEqual(result['test'][0]['label'], 0)

    def test_duplicate_between_splits_is_rejected(self):
        shutil.copyfile(self.root / 'train/NORMAL/0.png', self.root / 'test/NORMAL/0.png')
        with self.assertRaisesRegex(ValueError, 'multiple splits'):
            scan_dataset(self.root)

    def test_missing_class_is_rejected(self):
        shutil.rmtree(self.root / 'val/PNEUMONIA')
        with self.assertRaisesRegex(ValueError, 'Missing directory'):
            scan_dataset(self.root)

    def test_patient_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Patient.*multiple splits'):
            scan_dataset(self.root, r'^(\d+)')

class MetricTests(unittest.TestCase):
    def test_score_auc_and_recall(self):
        metrics = summarize([0, 0, 1, 1], [.1, .6, .8, .9])
        self.assertEqual(metrics['accuracy'], .75)
        self.assertEqual(metrics['roc_auc'], 1.0)
        self.assertEqual(metrics['pneumonia_recall'], 1.0)
        self.assertEqual(metrics['confusion_matrix'], [[1, 1], [0, 2]])

    def test_report_files(self):
        with tempfile.TemporaryDirectory() as temp:
            save_evaluation(temp, ['a', 'b'], [0, 1], [.1, .9])
            self.assertEqual(json.loads((Path(temp) / 'metrics.json').read_text())['samples'], 2)
            self.assertTrue((Path(temp) / 'evaluation.png').is_file())

if __name__ == '__main__':
    unittest.main()
