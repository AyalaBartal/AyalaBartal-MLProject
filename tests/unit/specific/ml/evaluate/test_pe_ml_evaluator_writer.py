import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock
import numpy as np

from src.common.plot import MlIoPlotWriter
from src.specific.ml.evaluate.pe_ml_evaluator_formatter import MlPeEvaluatorFormatter
from src.specific.ml.evaluate.pe_ml_file_writer import FileWriter
from src.specific.ml.evaluate.pe_ml_evaluator_writer import MlPeEvaluatorWriter
from src.specific.ml.evaluate.pe_ml_evaluate_report import MlPeEvaluateReport


class TestMlPeEvaluatorWriter(unittest.TestCase):

    def setUp(self):
        self.formatter = MagicMock(spec=MlPeEvaluatorFormatter)
        self.file_writer = MagicMock(spec=FileWriter)
        self.plot_writer = MagicMock(spec=MlIoPlotWriter)
        self.writer = MlPeEvaluatorWriter(
            self.formatter,
            self.file_writer,
            self.plot_writer
        )

    def test_init_stores_dependencies(self):
        self.assertIs(self.formatter, self.writer.formatter)
        self.assertIs(self.file_writer, self.writer.file_writer)
        self.assertIs(self.plot_writer, self.writer.plot_writer)

    def test_write_out_json_formats_and_writes(self):
        out_json = "out.json"
        threshold = 0.5
        report = SimpleNamespace(
            auc=0.91,
            acc=0.85,
            cm=np.array([[10, 2], [3, 15]])
        )
        json_data = {
            "auc": 0.91,
            "accuracy": 0.85,
            "threshold": 0.5,
            "confusion_matrix": [[10, 2], [3, 15]]
        }
        self.formatter.get_json_from_report_threshold.return_value = json_data

        self.writer.write_out_json(out_json, threshold, report)

        self.formatter.get_json_from_report_threshold.assert_called_once_with(report, threshold)
        self.file_writer.write_out_json.assert_called_once_with(out_json, json_data)

    def test_write_out_md_formats_and_writes(self):
        out_file = "out.md"
        threshold = 0.7
        report = SimpleNamespace(
            auc=0.93,
            acc=0.88,
            cm=np.array([[12, 1], [2, 18]])
        )
        md_text = "# PyTorch MLP — Test Metrics\n- AUC: 0.93"
        self.formatter.get_md_from_report_threshold.return_value = md_text

        self.writer.write_out_md(out_file, threshold, report)

        self.formatter.get_md_from_report_threshold.assert_called_once_with(report, threshold)
        self.file_writer.write_out_md.assert_called_once_with(out_file, md_text)

    def test_create_plot_of_confusion_matrix_calls_plot_writer(self):
        out_png = "confusion.png"
        report = SimpleNamespace(
            auc=0.91,
            acc=0.85,
            cm=np.array([[10, 2], [3, 15]])
        )

        self.writer.create_plot_of_confusion_matrix(out_png, report)

        self.plot_writer.create_plot.assert_called_once_with(out_png, report.cm)

    def test_write_out_json_with_different_threshold(self):
        out_json = "metrics.json"
        threshold = 0.9
        report = SimpleNamespace(auc=0.95, acc=0.90, cm=np.array([[20, 0], [0, 20]]))
        json_data = {"auc": 0.95, "accuracy": 0.90, "threshold": 0.9}
        self.formatter.get_json_from_report_threshold.return_value = json_data

        self.writer.write_out_json(out_json, threshold, report)

        self.file_writer.write_out_json.assert_called_once_with(out_json, json_data)
