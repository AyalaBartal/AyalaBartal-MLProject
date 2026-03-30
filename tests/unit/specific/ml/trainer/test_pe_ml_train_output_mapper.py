import unittest
from types import SimpleNamespace
import numpy as np

from src.specific.ml.trainer.pe_ml_train_output_mapper import MlPeTrainOutputMapper


class TestMlPeTrainOutputMapper(unittest.TestCase):

    def setUp(self):
        self.mapper = MlPeTrainOutputMapper()

    def test_map_report_to_output_returns_dict(self):
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

    def test_map_report_to_output_contains_metrics(self):
        report = {
            'cv_auc_mean': 0.85,
            'cv_auc_std': 0.02,
            'cv_accuracy_mean': 0.80,
            'cv_accuracy_std': 0.03,
            'confusion_matrix': [[100, 10], [15, 200]],
            'feature_names': ['f1', 'f2'],
            'model': None
        }
        train_result = SimpleNamespace()

        actual = self.mapper.map_report_to_output(report, train_result)

        self.assertIn('metrics', actual)
        self.assertIn('cv_auc_mean', actual['metrics'])
        self.assertIn('cv_auc_std', actual['metrics'])

    def test_map_report_to_output_contains_feature_schema(self):
        report = {
            'cv_auc_mean': 0.85,
            'cv_auc_std': 0.02,
            'cv_accuracy_mean': 0.80,
            'cv_accuracy_std': 0.03,
            'confusion_matrix': [],
            'feature_names': ['feat1', 'feat2', 'feat3'],
            'model': None
        }
        train_result = SimpleNamespace()

        actual = self.mapper.map_report_to_output(report, train_result)

        self.assertIn('feature_schema', actual)
        self.assertEqual(actual['feature_schema']['n_features'], 3)

    def test_map_report_to_output_feature_order_preserved(self):
        report = {
            'cv_auc_mean': 0.85,
            'cv_auc_std': 0.02,
            'cv_accuracy_mean': 0.80,
            'cv_accuracy_std': 0.03,
            'confusion_matrix': [],
            'feature_names': ['alpha', 'beta', 'gamma'],
            'model': None
        }
        train_result = SimpleNamespace()

        actual = self.mapper.map_report_to_output(report, train_result)

        self.assertEqual(actual['feature_schema']['feature_order'], ['alpha', 'beta', 'gamma'])

    def test_map_report_to_output_metrics_are_floats(self):
        report = {
            'cv_auc_mean': np.float64(0.85),
            'cv_auc_std': np.float64(0.02),
            'cv_accuracy_mean': np.float64(0.80),
            'cv_accuracy_std': np.float64(0.03),
            'confusion_matrix': [],
            'feature_names': ['f1'],
            'model': None
        }
        train_result = SimpleNamespace()

        actual = self.mapper.map_report_to_output(report, train_result)

        self.assertIsInstance(actual['metrics']['cv_auc_mean'], float)
        self.assertIsInstance(actual['metrics']['cv_auc_std'], float)

    def test_map_report_to_output_contains_model(self):
        model = object()
        report = {
            'cv_auc_mean': 0.85,
            'cv_auc_std': 0.02,
            'cv_accuracy_mean': 0.80,
            'cv_accuracy_std': 0.03,
            'confusion_matrix': [],
            'feature_names': ['f1'],
            'model': model
        }
        train_result = SimpleNamespace()

        actual = self.mapper.map_report_to_output(report, train_result)

        self.assertIn('model', actual)
        self.assertIs(actual['model'], model)
