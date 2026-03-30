import unittest
import numpy as np
import pandas as pd
from types import SimpleNamespace

from src.specific.ml.evaluate.pe_ml_evaluator_calculator import MlPeEvaluatorCalculator
from src.specific.ml.evaluate.pe_ml_evaluate_report import MlPeEvaluateReport


class MockModel:
    def __call__(self, x):
        return np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])


class TestMlPeEvaluatorCalculator(unittest.TestCase):

    def test_get_input_from_data_label_returns_df_without_label(self):
        args = SimpleNamespace(column_label="Label")
        df = pd.DataFrame({
            "f1": [1, 2, 3],
            "f2": [4, 5, 6],
            "Label": [0, 1, 0],
        })

        result = MlPeEvaluatorCalculator.get_input_from_data_label(args, df)

        self.assertNotIn("Label", result.columns)
        self.assertEqual(len(result.columns), 2)
        self.assertIn("f1", result.columns)
        self.assertIn("f2", result.columns)

    def test_get_output_from_data_label_returns_label_values(self):
        args = SimpleNamespace(column_label="Label")
        df = pd.DataFrame({
            "f1": [1, 2, 3],
            "Label": [0, 1, 0],
        })

        result = MlPeEvaluatorCalculator.get_output_from_data_label(args, df)

        self.assertEqual(len(result), 3)
        self.assertTrue(np.array_equal(result, np.array([0, 1, 0])))

    def test_get_output_from_data_label_returns_numpy_array(self):
        args = SimpleNamespace(column_label="target")
        df = pd.DataFrame({
            "f1": [1],
            "target": [1],
        })

        result = MlPeEvaluatorCalculator.get_output_from_data_label(args, df)

        self.assertIsInstance(result, np.ndarray)

    def test_get_pred_from_prob_threshold_returns_binary_predictions(self):
        proba = np.array([0.2, 0.5, 0.8, 0.1, 0.9])
        threshold = 0.5

        result = MlPeEvaluatorCalculator.get_pred_from_prob_threshold(proba, threshold)

        self.assertEqual(len(result), 5)
        self.assertTrue(np.array_equal(result, np.array([0, 1, 1, 0, 1])))

    def test_get_pred_from_prob_threshold_with_different_threshold(self):
        proba = np.array([0.3, 0.7])
        threshold = 0.6

        result = MlPeEvaluatorCalculator.get_pred_from_prob_threshold(proba, threshold)

        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 1)

    def test_get_report_from_y_prob_pred_returns_evaluate_report(self):
        y = np.array([0, 1, 0, 1, 1])
        prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
        pred = np.array([0, 1, 0, 1, 1])

        result = MlPeEvaluatorCalculator.get_report_from_y_prob_pred(y, prob, pred)

        self.assertIsInstance(result, MlPeEvaluateReport)

    def test_get_report_from_y_prob_pred_has_auc(self):
        y = np.array([0, 0, 1, 1])
        prob = np.array([0.1, 0.4, 0.6, 0.9])
        pred = np.array([0, 0, 1, 1])

        result = MlPeEvaluatorCalculator.get_report_from_y_prob_pred(y, prob, pred)

        self.assertTrue(hasattr(result, 'auc'))
        self.assertGreaterEqual(result.auc, 0.0)
        self.assertLessEqual(result.auc, 1.0)

    def test_get_report_from_y_prob_pred_has_accuracy(self):
        y = np.array([0, 1, 0, 1])
        prob = np.array([0.2, 0.8, 0.3, 0.9])
        pred = np.array([0, 1, 0, 1])

        result = MlPeEvaluatorCalculator.get_report_from_y_prob_pred(y, prob, pred)

        self.assertTrue(hasattr(result, 'acc'))
        self.assertEqual(result.acc, 1.0)

    def test_get_report_from_y_prob_pred_has_confusion_matrix(self):
        y = np.array([0, 1, 0, 1])
        prob = np.array([0.2, 0.8, 0.3, 0.9])
        pred = np.array([0, 1, 0, 1])

        result = MlPeEvaluatorCalculator.get_report_from_y_prob_pred(y, prob, pred)

        self.assertTrue(hasattr(result, 'cm'))
        self.assertIsNotNone(result.cm)
