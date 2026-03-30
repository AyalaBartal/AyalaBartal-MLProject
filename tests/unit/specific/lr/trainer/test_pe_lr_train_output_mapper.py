import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.specific.lr.trainer.pe_lr_train_output_mapper import LrPeTrainOutputMapper


class TestLrPeTrainOutputMapper(unittest.TestCase):

    def setUp(self):
        self.mapper = LrPeTrainOutputMapper()

    def test_map_report_to_output_returns_dict_with_structure(self):
        report = {
            'cv_auc_mean': 0.85,
            'cv_auc_std': 0.02,
            'cv_accuracy_mean': 0.80,
            'cv_accuracy_std': 0.03,
            'confusion_matrix': [[100, 10], [15, 200]],
            'feature_names': ['f1', 'f2', 'f3'],
            'model': None
        }
        train_result = SimpleNamespace()

        actual = self.mapper.map_report_to_output(report, train_result)

        self.assertIsInstance(actual, dict)
        self.assertIn('metrics', actual)
        self.assertIn('feature_schema', actual)
        self.assertIn('model', actual)
        self.assertAlmostEqual(actual['metrics']['cv_auc_mean'], 0.85)
        self.assertAlmostEqual(actual['metrics']['cv_accuracy_mean'], 0.80)
        self.assertEqual(actual['feature_schema']['n_features'], 3)

