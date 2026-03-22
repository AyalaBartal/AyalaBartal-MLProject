import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.specific.dt.evaluate import DtPeEvaluateReport
from src.specific.dt.evaluate.pe_dt_evaluator_calculator import DtPeEvaluatorCalculator


# We define two classes to simulate dt model and test method DtPeEvaluatorCalculator.get_prob_from_model_x.
# TestModelPredict with predict_proba and TestModelDecision with decision_function.
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


class TestDtPeEvaluatorCalculator(unittest.TestCase):

    def test_get_input_from_data_label_returns_df_without_label_column(self):
        args_al = SimpleNamespace(column_label="Label")
        df = pd.DataFrame({
            "f1": [1, 2, 3],
            "f2": [4, 5, 6],
            "Label": [0, 1, 0],
        })

        actual = DtPeEvaluatorCalculator.get_input_from_data_label(args_al, df)

        expected = pd.DataFrame({
            "f1": [1, 2, 3],
            "f2": [4, 5, 6],
        })

        pd.testing.assert_frame_equal(expected, actual)

    def test_get_output_from_data_label_returns_label_values(self):
        args_al = SimpleNamespace(column_label="Label")
        df = pd.DataFrame({
            "f1": [1, 2, 3],
            "Label": [0, 1, 0],
        })

        actual = DtPeEvaluatorCalculator.get_output_from_data_label(args_al, df)

        expected = np.array([0, 1, 0])

        np.testing.assert_array_equal(expected, actual)

    def test_get_prob_from_model_x_returns_positive_class_probability_when_predict_proba_exists(self):
        model = TestModelPredict()
        x = pd.DataFrame({"f1": [10, 20, 30]})

        actual = DtPeEvaluatorCalculator.get_prob_from_model_x(model, x)

        expected = np.array([0.1, 0.7, 0.8])

        np.testing.assert_array_almost_equal(expected, actual)

    def test_get_prob_from_model_x_normalizes_decision_function_when_predict_proba_not_exists(self):
        model = TestModelDecision()
        x = pd.DataFrame({"f1": [10, 20, 30]})

        actual = DtPeEvaluatorCalculator.get_prob_from_model_x(model, x)

        expected = np.array([0.0, 0.5, 1.0])

        np.testing.assert_array_almost_equal(expected, actual, decimal=6)

    def test_get_pred_from_prob_threshold_returns_binary_predictions(self):
        proba = np.array([0.2, 0.5, 0.7, 0.49, 0.51])
        threshold = 0.5

        actual = DtPeEvaluatorCalculator.get_pred_from_prob_threshold(proba, threshold)

        expected = np.array([0, 1, 1, 0, 1])

        np.testing.assert_array_equal(expected, actual)

    def test_get_report_from_y_prob_pred_returns_expected_report(self):
        y = np.array([0, 1, 1, 0])
        prob = np.array([0.1, 0.9, 0.8, 0.2])
        pred = np.array([0, 1, 1, 0])

        actual = DtPeEvaluatorCalculator.get_report_from_y_prob_pred(y, prob, pred)

        self.assertIsInstance(actual, DtPeEvaluateReport)
        self.assertAlmostEqual(1.0, actual.auc)
        self.assertAlmostEqual(1.0, actual.acc)
        np.testing.assert_array_equal(np.array([[2, 0], [0, 2]]), actual.cm)
