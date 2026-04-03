import unittest
from unittest.mock import MagicMock
import numpy as np

from src.specific.xgb.evaluate.pe_xgb_evaluator_writer import XgbPeEvaluatorWriter
from src.specific.xgb.evaluate.pe_xgb_evaluate_report import XgbPeEvaluateReport


class TestXgbPeEvaluatorWriter(unittest.TestCase):

    def setUp(self):
        self.formatter = MagicMock()
        self.file_writer = MagicMock()
        self.plot_writer = MagicMock()

        self.writer = XgbPeEvaluatorWriter(
            formatter=self.formatter,
            file_writer=self.file_writer,
            plot_writer=self.plot_writer
        )

    def test_init_sets_dependencies(self):
        self.assertIs(self.formatter, self.writer.formatter)
        self.assertIs(self.file_writer, self.writer.file_writer)
        self.assertIs(self.plot_writer, self.writer.plot_writer)

    def test_write_out_json_calls_formatter_and_file_writer(self):
        out_json = "output.json"
        threshold = 0.5
        report = XgbPeEvaluateReport(auc=0.92, acc=0.88, cm=np.array([[10, 2], [1, 12]]))

        json_data = {"auc": 0.92, "accuracy": 0.88, "threshold": 0.5}
        self.formatter.get_json_from_report_threshold.return_value = json_data

        self.writer.write_out_json(out_json, threshold, report)

        self.formatter.get_json_from_report_threshold.assert_called_once_with(report, threshold)

    def test_write_out_md_calls_formatter_and_file_writer(self):
        out_md = "output.md"
        threshold = 0.5
        report = XgbPeEvaluateReport(auc=0.92, acc=0.88, cm=np.array([[10, 2], [1, 12]]))

        md_text = "# XGBoost Evaluation Report\n\nAUC: 0.92\nAccuracy: 0.88"
        self.formatter.get_md_from_report_threshold.return_value = md_text

        self.writer.write_out_md(out_md, threshold, report)

        self.formatter.get_md_from_report_threshold.assert_called_once_with(report, threshold)

    def test_create_plot_of_confusion_matrix_calls_plot_writer(self):
        out_png = "confusion_matrix.png"
        report = XgbPeEvaluateReport(auc=0.92, acc=0.88, cm=np.array([[10, 2], [1, 12]]))

        self.writer.create_plot_of_confusion_matrix(out_png, report)

        self.plot_writer.write_confusion_matrix_plot.assert_called_once_with(out_png, report.cm)

    def test_write_out_json_with_different_threshold(self):
        out_json = "output.json"
        threshold = 0.7
        report = XgbPeEvaluateReport(auc=0.85, acc=0.82, cm=np.array([[8, 3], [2, 13]]))

        json_data = {"auc": 0.85, "accuracy": 0.82, "threshold": 0.7}
        self.formatter.get_json_from_report_threshold.return_value = json_data

        self.writer.write_out_json(out_json, threshold, report)

        call_args = self.formatter.get_json_from_report_threshold.call_args
        self.assertEqual(call_args[0][1], 0.7)


if __name__ == "__main__":
    unittest.main()


