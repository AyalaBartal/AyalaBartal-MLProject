import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.specific.cbst.evaluate import CbstPeEvaluateReport
from src.specific.lgb.evaluate.pe_lgb_evaluator_calculator import LgbPeEvaluatorCalculator


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


class TestCbstPeEvaluateReport(unittest.TestCase):

    def test_report_has_auc_attribute(self):
        report = CbstPeEvaluateReport(auc=0.92, acc=0.88, cm=np.array([[10, 2], [1, 12]]))

        self.assertEqual(report.auc, 0.92)

    def test_report_has_acc_attribute(self):
        report = CbstPeEvaluateReport(auc=0.92, acc=0.88, cm=np.array([[10, 2], [1, 12]]))

        self.assertEqual(report.acc, 0.88)

    def test_report_has_cm_attribute(self):
        cm = np.array([[10, 2], [1, 12]])
        report = CbstPeEvaluateReport(auc=0.92, acc=0.88, cm=cm)

        np.testing.assert_array_equal(report.cm, cm)

    def test_report_stores_different_auc_values(self):
        report1 = CbstPeEvaluateReport(auc=0.75, acc=0.80, cm=np.array([[5, 3], [2, 10]]))
        report2 = CbstPeEvaluateReport(auc=0.95, acc=0.92, cm=np.array([[10, 1], [1, 15]]))

        self.assertEqual(report1.auc, 0.75)
        self.assertEqual(report2.auc, 0.95)

    def test_report_stores_different_acc_values(self):
        report1 = CbstPeEvaluateReport(auc=0.75, acc=0.80, cm=np.array([[5, 3], [2, 10]]))
        report2 = CbstPeEvaluateReport(auc=0.95, acc=0.92, cm=np.array([[10, 1], [1, 15]]))

        self.assertEqual(report1.acc, 0.80)
        self.assertEqual(report2.acc, 0.92)

    def test_report_confusion_matrix_shape(self):
        cm = np.array([[10, 2], [1, 12]])
        report = CbstPeEvaluateReport(auc=0.92, acc=0.88, cm=cm)

        self.assertEqual(report.cm.shape, (2, 2))

    def test_report_with_perfect_scores(self):
        report = CbstPeEvaluateReport(auc=1.0, acc=1.0, cm=np.array([[10, 0], [0, 10]]))

        self.assertEqual(report.auc, 1.0)
        self.assertEqual(report.acc, 1.0)
        self.assertEqual(report.cm[0, 0], 10)
        self.assertEqual(report.cm[1, 1], 10)

    def test_report_with_poor_scores(self):
        report = CbstPeEvaluateReport(auc=0.5, acc=0.5, cm=np.array([[5, 5], [5, 5]]))

        self.assertEqual(report.auc, 0.5)
        self.assertEqual(report.acc, 0.5)


if __name__ == "__main__":
    unittest.main()
