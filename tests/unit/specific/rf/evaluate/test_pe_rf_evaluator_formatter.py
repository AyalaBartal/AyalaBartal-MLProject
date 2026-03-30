import unittest
from types import SimpleNamespace

import numpy as np

from src.specific.rf.evaluate import RfPeEvaluateReport
from src.specific.rf.evaluate.pe_rf_evaluator_formatter import RfPeEvaluatorFormatter


class TestRfPeEvaluatorFormatter(unittest.TestCase):

    def test_get_json_from_report_threshold_returns_dict(self):
        report = RfPeEvaluateReport(
            auc=0.85,
            acc=0.80,
            cm=np.array([[100, 10], [15, 200]])
        )

        actual = RfPeEvaluatorFormatter.get_json_from_report_threshold(report, 0.5)

        self.assertIsInstance(actual, dict)
        self.assertEqual(actual["auc"], 0.85)
        self.assertEqual(actual["accuracy"], 0.80)
        self.assertEqual(actual["threshold"], 0.5)
        self.assertIn("confusion_matrix", actual)

    def test_get_md_from_report_threshold_returns_string(self):
        report = RfPeEvaluateReport(
            auc=0.85,
            acc=0.80,
            cm=np.array([[100, 10], [15, 200]])
        )

        actual = RfPeEvaluatorFormatter.get_md_from_report_threshold(report, 0.5)

        self.assertIsInstance(actual, str)
        self.assertIn("AUC", actual)
        self.assertIn("Accuracy", actual)
