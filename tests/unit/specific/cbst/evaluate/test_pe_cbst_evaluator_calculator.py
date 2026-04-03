import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.specific.cbst.evaluate import CbstPeEvaluateReport
from src.specific.cbst.evaluate import CbstPeEvaluatorCalculator


class TestModelPredict:
    def predict_proba(self, x):
        return np.array([
            [0.9, 0.1],
            [0.3, 0.7],
            [0.2, 0.8],
        ])


class TestModelDecision:
    def decision_function(self, x):
        return np.array([-2.0, 0.0, 2.0])


class TestCbstPeEvaluatorCalculator(unittest.TestCase):

    def test_get_input_from_data_label_returns_df_without_label_column(self):
        args_al = SimpleNamespace(column_label="Label")
        df = pd.DataFrame({
            "f1": [1, 2, 3],
            "f2": [4, 5, 6],
            "Label": [0, 1, 0],
        })

        actual = CbstPeEvaluatorCalculator.get_input_from_data_label(args_al, df)

        expected = pd.DataFrame({
            "f1": [1, 2, 3],
            "f2": [4, 5, 6],
        })

        pd.testing.assert_frame_equal(expected, actual)

    def test_get_input_from_data_label_with_different_label_name(self):
        args_al = SimpleNamespace(column_label="target")
        df = pd.DataFrame({
            "feature1": [1, 2],
            "feature2": [3, 4],
            "target": [0, 1],
        })

        actual = CbstPeEvaluatorCalculator.get_input_from_data_label(args_al, df)

        self.assertNotIn("target", actual.columns)
        self.assertEqual(len(actual.columns), 2)

    def test_get_output_from_data_label_returns_label_values(self):
        args_al = SimpleNamespace(column_label="Label")
        df = pd.DataFrame({
            "f1": [1, 2, 3],
            "Label": [0, 1, 0],
        })

        actual = CbstPeEvaluatorCalculator.get_output_from_data_label(args_al, df)

        expected = np.array([0, 1, 0])

        np.testing.assert_array_equal(expected, actual)

    def test_get_output_from_data_label_different_label_column(self):
        args_al = SimpleNamespace(column_label="is_positive")
        df = pd.DataFrame({
            "x": [1, 2, 3, 4],
            "is_positive": [1, 0, 1, 0],
        })

        actual = CbstPeEvaluatorCalculator.get_output_from_data_label(args_al, df)

        expected = np.array([1, 0, 1, 0])
        np.testing.assert_array_equal(expected, actual)

    def test_get_prob_from_model_x_returns_positive_class_probability_when_predict_proba_exists(self):
        model = TestModelPredict()
        x = pd.DataFrame({"f1": [10, 20, 30]})

        actual = CbstPeEvaluatorCalculator.get_prob_from_model_x(model, x)

        expected = np.array([0.1, 0.7, 0.8])

        np.testing.assert_array_almost_equal(expected, actual)

    def test_get_prob_from_model_x_normalizes_decision_function_when_predict_proba_not_exists(self):
        model = TestModelDecision()
        x = pd.DataFrame({"f1": [10, 20, 30]})

        actual = CbstPeEvaluatorCalculator.get_prob_from_model_x(model, x)

        expected = np.array([0.0, 0.5, 1.0])

        np.testing.assert_array_almost_equal(expected, actual, decimal=6)

    def test_get_prob_from_model_x_handles_large_decision_values(self):
        model = TestModelDecision()
        x = pd.DataFrame({"f1": [1, 2, 3]})

        actual = CbstPeEvaluatorCalculator.get_prob_from_model_x(model, x)

        self.assertTrue(np.all(actual >= 0.0))
        self.assertTrue(np.all(actual <= 1.0))

    def test_get_pred_from_prob_threshold_returns_binary_predictions(self):
        proba = np.array([0.2, 0.5, 0.7, 0.49, 0.51])
        threshold = 0.5

        actual = CbstPeEvaluatorCalculator.get_pred_from_prob_threshold(proba, threshold)

        expected = np.array([0, 1, 1, 0, 1])

        np.testing.assert_array_equal(expected, actual)

    def test_get_pred_from_prob_threshold_with_different_threshold(self):
        proba = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        threshold = 0.6

        actual = CbstPeEvaluatorCalculator.get_pred_from_prob_threshold(proba, threshold)

        expected = np.array([0, 0, 0, 1, 1])

        np.testing.assert_array_equal(expected, actual)

    def test_get_pred_from_prob_threshold_all_below_threshold(self):
        proba = np.array([0.1, 0.2, 0.3])
        threshold = 0.5

        actual = CbstPeEvaluatorCalculator.get_pred_from_prob_threshold(proba, threshold)

        expected = np.array([0, 0, 0])

        np.testing.assert_array_equal(expected, actual)

    def test_get_pred_from_prob_threshold_all_above_threshold(self):
        proba = np.array([0.6, 0.7, 0.8])
        threshold = 0.5

        actual = CbstPeEvaluatorCalculator.get_pred_from_prob_threshold(proba, threshold)

        expected = np.array([1, 1, 1])

        np.testing.assert_array_equal(expected, actual)

    def test_get_report_from_y_prob_pred_returns_expected_report(self):
        y = np.array([0, 1, 1, 0])
        prob = np.array([0.1, 0.9, 0.8, 0.2])
        pred = np.array([0, 1, 1, 0])

        actual = CbstPeEvaluatorCalculator.get_report_from_y_prob_pred(y, prob, pred)

        self.assertIsInstance(actual, CbstPeEvaluateReport)
        self.assertAlmostEqual(1.0, actual.auc)
        self.assertAlmostEqual(1.0, actual.acc)
        np.testing.assert_array_equal(np.array([[2, 0], [0, 2]]), actual.cm)

    def test_get_report_from_y_prob_pred_with_some_errors(self):
        y = np.array([0, 1, 0, 1])
        prob = np.array([0.2, 0.8, 0.3, 0.9])
        pred = np.array([0, 1, 0, 1])

        actual = CbstPeEvaluatorCalculator.get_report_from_y_prob_pred(y, prob, pred)

        self.assertEqual(actual.acc, 1.0)
        np.testing.assert_array_equal(actual.cm, np.array([[2, 0], [0, 2]]))

    def test_get_report_from_y_prob_pred_calculates_auc(self):
        y = np.array([0, 0, 1, 1])
        prob = np.array([0.1, 0.4, 0.35, 0.8])
        pred = np.array([0, 0, 0, 1])

        actual = CbstPeEvaluatorCalculator.get_report_from_y_prob_pred(y, prob, pred)

        self.assertGreater(actual.auc, 0.0)
        self.assertLessEqual(actual.auc, 1.0)

    def test_get_report_from_y_prob_pred_calculates_confusion_matrix(self):
        y = np.array([0, 0, 1, 1])
        prob = np.array([0.2, 0.3, 0.7, 0.8])
        pred = np.array([0, 0, 1, 1])

        actual = CbstPeEvaluatorCalculator.get_report_from_y_prob_pred(y, prob, pred)

        self.assertEqual(actual.cm.shape, (2, 2))
        self.assertEqual(actual.cm[0, 0], 2)
        self.assertEqual(actual.cm[1, 1], 2)

    def test_get_report_from_y_prob_pred_with_misclassifications(self):
        y = np.array([0, 1, 0, 1])
        prob = np.array([0.9, 0.1, 0.8, 0.2])
        pred = np.array([1, 0, 1, 0])

        actual = CbstPeEvaluatorCalculator.get_report_from_y_prob_pred(y, prob, pred)

        self.assertEqual(actual.acc, 0.0)
        np.testing.assert_array_equal(actual.cm, np.array([[0, 2], [2, 0]]))


if __name__ == "__main__":
    unittest.main()

