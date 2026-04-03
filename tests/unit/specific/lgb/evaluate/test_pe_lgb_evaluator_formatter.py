import unittest
from types import SimpleNamespace
import numpy as np

from src.specific.lgb.evaluate.pe_lgb_evaluator_formatter import LgbPeEvaluatorFormatter
from src.specific.lgb.evaluate.pe_lgb_evaluate_report import LgbPeEvaluateReport


class TestLgbPeEvaluatorFormatter(unittest.TestCase):

    def test_get_json_from_report_threshold_returns_dict(self):
        report = LgbPeEvaluateReport(auc=0.92, acc=0.88, cm=np.array([[10, 2], [1, 12]]))
        threshold = 0.5

        actual = LgbPeEvaluatorFormatter.get_json_from_report_threshold(report, threshold)

        self.assertIsInstance(actual, dict)

    def test_get_json_from_report_threshold_contains_auc(self):
        report = LgbPeEvaluateReport(auc=0.92, acc=0.88, cm=np.array([[10, 2], [1, 12]]))
        threshold = 0.5

        actual = LgbPeEvaluatorFormatter.get_json_from_report_threshold(report, threshold)

        self.assertIn("auc", actual)
        self.assertAlmostEqual(actual["auc"], 0.92, places=5)

    def test_get_json_from_report_threshold_contains_accuracy(self):
        report = LgbPeEvaluateReport(auc=0.92, acc=0.88, cm=np.array([[10, 2], [1, 12]]))
        threshold = 0.5

        actual = LgbPeEvaluatorFormatter.get_json_from_report_threshold(report, threshold)

        self.assertIn("accuracy", actual)
        self.assertAlmostEqual(actual["accuracy"], 0.88, places=5)

    def test_get_json_from_report_threshold_contains_threshold(self):
        report = LgbPeEvaluateReport(auc=0.92, acc=0.88, cm=np.array([[10, 2], [1, 12]]))
        threshold = 0.7

        actual = LgbPeEvaluatorFormatter.get_json_from_report_threshold(report, threshold)

        self.assertIn("threshold", actual)
        self.assertEqual(actual["threshold"], 0.7)

    def test_get_json_from_report_threshold_contains_confusion_matrix(self):
        report = LgbPeEvaluateReport(auc=0.92, acc=0.88, cm=np.array([[10, 2], [1, 12]]))
        threshold = 0.5

        actual = LgbPeEvaluatorFormatter.get_json_from_report_threshold(report, threshold)

        self.assertIn("confusion_matrix", actual)

    def test_get_json_from_report_threshold_with_different_values(self):
        report = LgbPeEvaluateReport(auc=0.75, acc=0.80, cm=np.array([[5, 3], [2, 10]]))
        threshold = 0.6

        actual = LgbPeEvaluatorFormatter.get_json_from_report_threshold(report, threshold)

        self.assertAlmostEqual(actual["auc"], 0.75, places=5)
        self.assertAlmostEqual(actual["accuracy"], 0.80, places=5)
        self.assertEqual(actual["threshold"], 0.6)

    def test_get_md_from_report_threshold_returns_string(self):
        report = LgbPeEvaluateReport(auc=0.92, acc=0.88, cm=np.array([[10, 2], [1, 12]]))
        threshold = 0.5

        actual = LgbPeEvaluatorFormatter.get_md_from_report_threshold(report, threshold)

        self.assertIsInstance(actual, str)

    def test_get_md_from_report_threshold_contains_auc(self):
        report = LgbPeEvaluateReport(auc=0.92, acc=0.88, cm=np.array([[10, 2], [1, 12]]))
        threshold = 0.5

        actual = LgbPeEvaluatorFormatter.get_md_from_report_threshold(report, threshold)

        self.assertIn("AUC", actual)
        self.assertIn("0.92", actual)

    def test_get_md_from_report_threshold_contains_accuracy(self):
        report = LgbPeEvaluateReport(auc=0.92, acc=0.88, cm=np.array([[10, 2], [1, 12]]))
        threshold = 0.5

        actual = LgbPeEvaluatorFormatter.get_md_from_report_threshold(report, threshold)

        self.assertIn("Accuracy", actual)
        self.assertIn("0.88", actual)

    def test_get_md_from_report_threshold_contains_threshold(self):
        report = LgbPeEvaluateReport(auc=0.92, acc=0.88, cm=np.array([[10, 2], [1, 12]]))
        threshold = 0.7

        actual = LgbPeEvaluatorFormatter.get_md_from_report_threshold(report, threshold)

        self.assertIn("0.70", actual)

    def test_get_md_from_report_threshold_with_high_scores(self):
        report = LgbPeEvaluateReport(auc=0.99, acc=0.97, cm=np.array([[100, 1], [2, 97]]))
        threshold = 0.5

        actual = LgbPeEvaluatorFormatter.get_md_from_report_threshold(report, threshold)

        self.assertIn("0.99", actual)
        self.assertIn("0.97", actual)

    def test_get_md_from_report_threshold_with_low_scores(self):
        report = LgbPeEvaluateReport(auc=0.51, acc=0.52, cm=np.array([[26, 24], [24, 26]]))
        threshold = 0.5

        actual = LgbPeEvaluatorFormatter.get_md_from_report_threshold(report, threshold)

        self.assertIn("0.51", actual)
        self.assertIn("0.52", actual)


if __name__ == "__main__":
    unittest.main()
