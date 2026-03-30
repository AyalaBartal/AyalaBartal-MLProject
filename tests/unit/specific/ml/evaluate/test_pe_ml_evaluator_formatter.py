import unittest
from types import SimpleNamespace
import numpy as np

from src.specific.ml.evaluate.pe_ml_evaluator_formatter import MlPeEvaluatorFormatter
from src.specific.ml.evaluate.pe_ml_evaluate_report import MlPeEvaluateReport


class TestMlPeEvaluatorFormatter(unittest.TestCase):

    def test_get_json_from_report_threshold_returns_dict(self):
        report = MlPeEvaluateReport(
            auc=0.85,
            acc=0.80,
            cm=np.array([[100, 10], [15, 200]])
        )

        result = MlPeEvaluatorFormatter.get_json_from_report_threshold(report, 0.5)

        self.assertIsInstance(result, dict)

    def test_get_json_from_report_threshold_contains_auc(self):
        report = MlPeEvaluateReport(
            auc=0.85,
            acc=0.80,
            cm=np.array([[100, 10], [15, 200]])
        )

        result = MlPeEvaluatorFormatter.get_json_from_report_threshold(report, 0.5)

        self.assertEqual(result["auc"], 0.85)

    def test_get_json_from_report_threshold_contains_accuracy(self):
        report = MlPeEvaluateReport(
            auc=0.85,
            acc=0.80,
            cm=np.array([[100, 10], [15, 200]])
        )

        result = MlPeEvaluatorFormatter.get_json_from_report_threshold(report, 0.5)

        self.assertEqual(result["accuracy"], 0.80)

    def test_get_json_from_report_threshold_contains_threshold(self):
        report = MlPeEvaluateReport(
            auc=0.85,
            acc=0.80,
            cm=np.array([[100, 10], [15, 200]])
        )

        result = MlPeEvaluatorFormatter.get_json_from_report_threshold(report, 0.7)

        self.assertEqual(result["threshold"], 0.7)

    def test_get_json_from_report_threshold_contains_confusion_matrix(self):
        cm = np.array([[100, 10], [15, 200]])
        report = MlPeEvaluateReport(auc=0.85, acc=0.80, cm=cm)

        result = MlPeEvaluatorFormatter.get_json_from_report_threshold(report, 0.5)

        self.assertEqual(result["confusion_matrix"], [[100, 10], [15, 200]])

    def test_get_md_from_report_threshold_returns_string(self):
        report = MlPeEvaluateReport(
            auc=0.85,
            acc=0.80,
            cm=np.array([[100, 10], [15, 200]])
        )

        result = MlPeEvaluatorFormatter.get_md_from_report_threshold(report, 0.5)

        self.assertIsInstance(result, str)

    def test_get_md_from_report_threshold_contains_auc(self):
        report = MlPeEvaluateReport(
            auc=0.85,
            acc=0.80,
            cm=np.array([[100, 10], [15, 200]])
        )

        result = MlPeEvaluatorFormatter.get_md_from_report_threshold(report, 0.5)

        self.assertIn("AUC", result)
        self.assertIn("0.85", result)

    def test_get_md_from_report_threshold_contains_accuracy(self):
        report = MlPeEvaluateReport(
            auc=0.85,
            acc=0.80,
            cm=np.array([[100, 10], [15, 200]])
        )

        result = MlPeEvaluatorFormatter.get_md_from_report_threshold(report, 0.5)

        self.assertIn("Accuracy", result)
        self.assertIn("0.80", result)

    def test_get_md_from_report_threshold_contains_threshold(self):
        report = MlPeEvaluateReport(
            auc=0.85,
            acc=0.80,
            cm=np.array([[100, 10], [15, 200]])
        )

        result = MlPeEvaluatorFormatter.get_md_from_report_threshold(report, 0.7)

        self.assertIn("Threshold", result)
        self.assertIn("0.70", result)
