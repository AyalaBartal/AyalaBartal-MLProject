import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from src.common.plot import MlIoPlotWriter
from src.specific.rf.evaluate.pe_rf_evaluator_formatter import RfPeEvaluatorFormatter
from src.specific.rf.evaluate.pe_rf_file_writer import FileWriter
from src.specific.rf.evaluate.pe_rf_evaluator_writer import RfPeEvaluatorWriter


class TestRfPeEvaluatorWriter(unittest.TestCase):

    def setUp(self):
        self.formatter = MagicMock(spec=RfPeEvaluatorFormatter)
        self.file_writer = MagicMock(spec=FileWriter)
        self.plot_writer = MagicMock(spec=MlIoPlotWriter)
        self.writer = RfPeEvaluatorWriter(
            self.formatter,
            self.file_writer,
            self.plot_writer
        )

    def test_write_out_json_formats_data_and_writes_file(self):
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

    def test_write_out_md_formats_text_and_writes_file(self):
        out_file = "out.md"
        threshold = 0.7
        report = SimpleNamespace(
            auc=0.93,
            acc=0.88,
            cm=np.array([[8, 1], [2, 20]])
        )
        out_text = "\n".join([
            "# Random Forest — Test Metrics",
            "- AUC: 0.9300",
            "- Accuracy: 0.8800",
            "- Threshold: 0.70",
            "- Confusion: [[8, 1], [2, 20]]",
        ])
        self.formatter.get_md_from_report_threshold.return_value = out_text

        self.writer.write_out_md(out_file, threshold, report)

        self.formatter.get_md_from_report_threshold.assert_called_once_with(report, threshold)
        self.file_writer.write_out_md.assert_called_once_with(out_file, out_text)

    def test_create_plot_of_confusion_matrix_calls_plot_writer_with_cm(self):
        out_png = "out.png"
        cm = np.array([[12, 4], [1, 9]])
        report = SimpleNamespace(cm=cm)

        self.writer.create_plot_of_confusion_matrix(out_png, report)

        self.plot_writer.create_plot.assert_called_once_with(out_png, cm)

    def test_init_sets_dependencies(self):
        formatter = MagicMock(spec=RfPeEvaluatorFormatter)
        file_writer = MagicMock(spec=FileWriter)
        plot_writer = MagicMock(spec=MlIoPlotWriter)

        writer = RfPeEvaluatorWriter(formatter, file_writer, plot_writer)

        self.assertIs(formatter, writer.formatter)
        self.assertIs(file_writer, writer.file_writer)
        self.assertIs(plot_writer, writer.plot_writer)
