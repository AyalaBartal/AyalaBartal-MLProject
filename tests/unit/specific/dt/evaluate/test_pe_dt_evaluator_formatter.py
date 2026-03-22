import unittest
from types import SimpleNamespace
import numpy as np

from src.specific.dt.evaluate import DtPeEvaluatorFormatter


class TestDtPeEvaluatorFormatter(unittest.TestCase):

    def test_get_json_from_report_threshold_returns_expected_dict(self):
        report = SimpleNamespace(
            auc=np.float64(0.91234),
            acc=np.float64(0.85678),
            cm=np.array([[10, 2], [3, 15]])
        )
        threshold = 0.5

        actual = DtPeEvaluatorFormatter.get_json_from_report_threshold(report, threshold)

        expected = {
            'auc': 0.91234,
            'accuracy': 0.85678,
            'threshold': 0.5,
            'confusion_matrix': [[10, 2], [3, 15]]
        }

        self.assertEqual(expected, actual)
        self.assertIsInstance(actual["auc"], float)
        self.assertIsInstance(actual["accuracy"], float)
        self.assertIsInstance(actual["confusion_matrix"], list)

    def test_get_md_from_report_threshold_returns_expected_markdown(self):
        report = SimpleNamespace(
            auc=0.91234,
            acc=0.85678,
            cm=np.array([[10, 2], [3, 15]])
        )
        threshold = 0.5

        actual = DtPeEvaluatorFormatter.get_md_from_report_threshold(report, threshold)

        expected = "\n".join([
            "# Decision Tree — Test Metrics",
            "- AUC: 0.9123",
            "- Accuracy: 0.8568",
            "- Threshold: 0.50",
            "- Confusion: [[10, 2], [3, 15]]",
        ])

        self.assertEqual(expected, actual)

    def test_get_md_from_report_threshold_formats_values_to_expected_precision(self):
        report = SimpleNamespace(
            auc=0.1,
            acc=0.98765,
            cm=np.array([[1, 0], [0, 1]])
        )
        threshold = 0.3333

        actual = DtPeEvaluatorFormatter.get_md_from_report_threshold(report, threshold)

        self.assertIn("- AUC: 0.1000", actual)
        self.assertIn("- Accuracy: 0.9877", actual)
        self.assertIn("- Threshold: 0.33", actual)

    def test_get_json_from_report_threshold_uses_given_threshold_argument(self):
        report = SimpleNamespace(
            auc=0.8,
            acc=0.9,
            cm=np.array([[1, 2], [3, 4]])
        )
        threshold = 0.75

        actual = DtPeEvaluatorFormatter.get_json_from_report_threshold(report, threshold)

        self.assertEqual(0.75, actual["threshold"])
