import unittest
import pandas as pd
import numpy as np

from src.specific.ml.trainer.pe_ml_report_trainer import MlPeReportTrainer


class TestMlPeReportTrainer(unittest.TestCase):

    def setUp(self):
        self.trainer = MlPeReportTrainer()

    def test_get_report_returns_dict_with_required_keys(self):
        from unittest.mock import Mock
        args = Mock()
        ml_features = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
        model = Mock()
        cv_results = {
            'test_auc': np.array([0.8, 0.85]),
            'test_accuracy': np.array([0.7, 0.75])
        }

        result = self.trainer.get_report(args, ml_features, model, cv_results)

        self.assertIsInstance(result, dict)
        self.assertIn('cv_auc_mean', result)
        self.assertIn('cv_auc_std', result)
        self.assertIn('cv_accuracy_mean', result)
        self.assertIn('cv_accuracy_std', result)
        self.assertIn('confusion_matrix', result)
        self.assertIn('model', result)
        self.assertIn('feature_names', result)

    def test_get_report_calculates_correct_auc_mean(self):
        from unittest.mock import Mock
        args = Mock()
        ml_features = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
        model = Mock()
        cv_results = {
            'test_auc': np.array([0.8, 0.9]),
            'test_accuracy': np.array([0.7, 0.75])
        }

        result = self.trainer.get_report(args, ml_features, model, cv_results)

        self.assertAlmostEqual(result['cv_auc_mean'], 0.85, places=2)

    def test_get_report_calculates_correct_accuracy_std(self):
        from unittest.mock import Mock
        args = Mock()
        ml_features = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
        model = Mock()
        cv_results = {
            'test_auc': np.array([0.8]),
            'test_accuracy': np.array([0.75])
        }

        result = self.trainer.get_report(args, ml_features, model, cv_results)

        self.assertEqual(result['cv_accuracy_std'], 0.0)

    def test_get_report_preserves_feature_names(self):
        from unittest.mock import Mock
        args = Mock()
        ml_features = pd.DataFrame({"feat_a": [1], "feat_b": [2], "feat_c": [3]})
        model = Mock()
        cv_results = {
            'test_auc': np.array([0.8]),
            'test_accuracy': np.array([0.7])
        }

        result = self.trainer.get_report(args, ml_features, model, cv_results)

        self.assertEqual(result['feature_names'], ['feat_a', 'feat_b', 'feat_c'])

    def test_get_report_stores_model_reference(self):
        from unittest.mock import Mock
        args = Mock()
        ml_features = pd.DataFrame({"f1": [1, 2]})
        model = Mock(name="test_model")
        cv_results = {
            'test_auc': np.array([0.8]),
            'test_accuracy': np.array([0.7])
        }

        result = self.trainer.get_report(args, ml_features, model, cv_results)

        self.assertIs(result['model'], model)

    def test_build_confusion_matrix_with_perfect_predictions(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]

        result = self.trainer._build_confusion_matrix(y_true, y_pred)

        self.assertEqual(result['true_neg'], 2)
        self.assertEqual(result['true_pos'], 2)
        self.assertEqual(result['false_pos'], 0)
        self.assertEqual(result['false_neg'], 0)

    def test_build_confusion_matrix_with_errors(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]

        result = self.trainer._build_confusion_matrix(y_true, y_pred)

        self.assertEqual(result['true_neg'], 1)
        self.assertEqual(result['false_pos'], 1)
        self.assertEqual(result['false_neg'], 1)
        self.assertEqual(result['true_pos'], 1)

    def test_build_confusion_matrix_with_all_positives(self):
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 1, 1]

        result = self.trainer._build_confusion_matrix(y_true, y_pred)

        self.assertEqual(result['true_neg'], 0)
        self.assertEqual(result['false_pos'], 2)
        self.assertEqual(result['false_neg'], 0)
        self.assertEqual(result['true_pos'], 2)
