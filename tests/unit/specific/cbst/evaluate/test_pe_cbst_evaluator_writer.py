import unittest
from unittest.mock import MagicMock

import numpy as np

from src.specific.cbst.evaluate import CbstPeEvaluateReport
from src.specific.cbst.evaluate import CbstPeEvaluatorWriter


class TestCbstPeEvaluatorWriter(unittest.TestCase):

    def setUp(self):
        self.formatter = MagicMock()
        self.file_writer = MagicMock()
        self.plot_writer = MagicMock()

        self.writer = CbstPeEvaluatorWriter(
            formatter=self.formatter,
            file_writer=self.file_writer,
            plot_writer=self.plot_writer
        )

    def test_init_sets_dependencies(self):
        self.assertIs(self.formatter, self.writer.formatter)
        self.assertIs(self.file_writer, self.writer.file_writer)
        self.assertIs(self.plot_writer, self.writer.plot_writer)

    def test_write_out_json_calls_formatter_and_file_writer(self):
        out_json_path = "/output/report.json"
        threshold = 0.5
        report = CbstPeEvaluateReport(auc=0.92, acc=0.88, cm=np.array([[10, 2], [1, 12]]))
        formatted_json = {"auc": 0.92, "accuracy": 0.88}

        self.formatter.get_json_from_report_threshold.return_value = formatted_json

        self.writer.write_out_json(out_json_path, threshold, report)

        self.formatter.get_json_from_report_threshold.assert_called_once_with(report, threshold)
        self.file_writer.write_out_json.assert_called_once_with(out_json_path, formatted_json)

    def test_write_out_md_calls_formatter_and_file_writer(self):
        out_md_path = "/output/report.md"
        threshold = 0.5
        report = CbstPeEvaluateReport(auc=0.92, acc=0.88, cm=np.array([[10, 2], [1, 12]]))
        formatted_md = "# Report\nAUC: 0.92"

        self.formatter.get_md_from_report_threshold.return_value = formatted_md

        self.writer.write_out_md(out_md_path, threshold, report)

        self.formatter.get_md_from_report_threshold.assert_called_once_with(report, threshold)
        self.file_writer.write_out_md.assert_called_once_with(out_md_path, formatted_md)

    def create_plot_of_confusion_matrix_calls_plot_writer(self):
        out_png_path = "/output/confusion_matrix.png"
        report = CbstPeEvaluateReport(auc=0.92, acc=0.88, cm=np.array([[10, 2], [1, 12]]))

        self.writer.create_plot_of_confusion_matrix(out_png_path, report)

        self.plot_writer.create_plot.assert_called_once_with(out_png_path, report.cm)

    def test_write_out_json_with_different_paths(self):
        out_json_path = "/custom/path/results.json"
        threshold = 0.7
        report = CbstPeEvaluateReport(auc=0.95, acc=0.93, cm=np.array([[15, 1], [2, 20]]))
        formatted_json = {"auc": 0.95}

        self.formatter.get_json_from_report_threshold.return_value = formatted_json

        self.writer.write_out_json(out_json_path, threshold, report)

        self.file_writer.write_out_json.assert_called_once_with(out_json_path, formatted_json)

    def test_write_out_md_with_different_paths(self):
        out_md_path = "/custom/path/results.md"
        threshold = 0.7
        report = CbstPeEvaluateReport(auc=0.95, acc=0.93, cm=np.array([[15, 1], [2, 20]]))
        formatted_md = "# Results"

        self.formatter.get_md_from_report_threshold.return_value = formatted_md

        self.writer.write_out_md(out_md_path, threshold, report)

        self.file_writer.write_out_md.assert_called_once_with(out_md_path, formatted_md)


if __name__ == "__main__":
    unittest.main()
