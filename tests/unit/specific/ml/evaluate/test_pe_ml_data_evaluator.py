import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock
import pandas as pd
import numpy as np

from src.specific.ml.evaluate.pe_ml_evaluator import MlPeDataEvaluator
from src.specific.ml.evaluate.pe_ml_evaluator_reader import MlPeEvaluatorReader
from src.specific.ml.evaluate.pe_ml_evaluator_calculator import MlPeEvaluatorCalculator
from src.specific.ml.evaluate.pe_ml_evaluator_writer import MlPeEvaluatorWriter


class TestMlPeDataEvaluator(unittest.TestCase):

    def setUp(self):
        self.reader = MagicMock(spec=MlPeEvaluatorReader)
        self.calculator = MagicMock(spec=MlPeEvaluatorCalculator)
        self.writer = MagicMock(spec=MlPeEvaluatorWriter)

        self.evaluator = MlPeDataEvaluator(
            reader=self.reader,
            calculator=self.calculator,
            writer=self.writer
        )

    def test_init_stores_dependencies(self):
        self.assertIs(self.reader, self.evaluator.reader)
        self.assertIs(self.calculator, self.evaluator.calculator)
        self.assertIs(self.writer, self.evaluator.writer)

    def test_evaluate_validates_input_and_output(self):
        args_in = SimpleNamespace(input_csv="test.csv", input_model="model.pt", input_dir="/input")
        args_al = SimpleNamespace(column_label="label", threshold=0.5)
        args_out = SimpleNamespace(output_dir="/output", out_json="out.json", out_md="out.md", out_png="out.png")

        self.reader.read_csv_to_df.return_value = pd.DataFrame()
        self.reader.read_ml_model_from_pt_file.return_value = MagicMock()
        self.calculator.get_output_from_data_label.return_value = np.array([])
        self.calculator.get_input_from_data_label.return_value = pd.DataFrame()
        self.calculator.get_prob_from_model_x.return_value = np.array([])
        self.calculator.get_pred_from_prob_threshold.return_value = np.array([])
        self.calculator.get_report_from_y_prob_pred.return_value = MagicMock()

        self.evaluator.evaluate(args_in, args_al, args_out)

        self.reader.validate_input.assert_called_once_with(args_in)
        self.reader.validate_output.assert_called_once_with(args_out)

    def test_evaluate_reads_csv(self):
        args_in = SimpleNamespace(input_csv="test.csv", input_model="model.pt", input_dir="/input")
        args_al = SimpleNamespace(column_label="label", threshold=0.5)
        args_out = SimpleNamespace(output_dir="/output", out_json="out.json", out_md="out.md", out_png="out.png")

        df = pd.DataFrame()
        self.reader.read_csv_to_df.return_value = df
        self.reader.read_ml_model_from_pt_file.return_value = MagicMock()
        self.calculator.get_output_from_data_label.return_value = np.array([])
        self.calculator.get_input_from_data_label.return_value = pd.DataFrame()
        self.calculator.get_prob_from_model_x.return_value = np.array([])
        self.calculator.get_pred_from_prob_threshold.return_value = np.array([])
        self.calculator.get_report_from_y_prob_pred.return_value = MagicMock()

        self.evaluator.evaluate(args_in, args_al, args_out)

        self.reader.read_csv_to_df.assert_called_once_with(args_in.input_csv)

    def test_evaluate_reads_model(self):
        args_in = SimpleNamespace(input_csv="test.csv", input_model="model.pt", input_dir="/input")
        args_al = SimpleNamespace(column_label="label", threshold=0.5)
        args_out = SimpleNamespace(output_dir="/output", out_json="out.json", out_md="out.md", out_png="out.png")

        self.reader.read_csv_to_df.return_value = pd.DataFrame()
        self.reader.read_ml_model_from_pt_file.return_value = MagicMock()
        self.calculator.get_output_from_data_label.return_value = np.array([])
        self.calculator.get_input_from_data_label.return_value = pd.DataFrame()
        self.calculator.get_prob_from_model_x.return_value = np.array([])
        self.calculator.get_pred_from_prob_threshold.return_value = np.array([])
        self.calculator.get_report_from_y_prob_pred.return_value = MagicMock()

        self.evaluator.evaluate(args_in, args_al, args_out)

        self.reader.read_ml_model_from_pt_file.assert_called_once_with(args_in)

    def test_evaluate_calculates_probabilities(self):
        args_in = SimpleNamespace(input_csv="test.csv", input_model="model.pt", input_dir="/input")
        args_al = SimpleNamespace(column_label="label", threshold=0.5)
        args_out = SimpleNamespace(output_dir="/output", out_json="out.json", out_md="out.md", out_png="out.png")

        df = pd.DataFrame({"f1": [1, 2]})
        model = MagicMock()
        self.reader.read_csv_to_df.return_value = df
        self.reader.read_ml_model_from_pt_file.return_value = model
        self.calculator.get_output_from_data_label.return_value = np.array([])
        self.calculator.get_input_from_data_label.return_value = pd.DataFrame()
        self.calculator.get_prob_from_model_x.return_value = np.array([])
        self.calculator.get_pred_from_prob_threshold.return_value = np.array([])
        self.calculator.get_report_from_y_prob_pred.return_value = MagicMock()

        self.evaluator.evaluate(args_in, args_al, args_out)

        self.calculator.get_prob_from_model_x.assert_called_once()

    def test_evaluate_calculates_predictions(self):
        args_in = SimpleNamespace(input_csv="test.csv", input_model="model.pt", input_dir="/input")
        args_al = SimpleNamespace(column_label="label", threshold=0.5)
        args_out = SimpleNamespace(output_dir="/output", out_json="out.json", out_md="out.md", out_png="out.png")

        self.reader.read_csv_to_df.return_value = pd.DataFrame()
        self.reader.read_ml_model_from_pt_file.return_value = MagicMock()
        self.calculator.get_output_from_data_label.return_value = np.array([])
        self.calculator.get_input_from_data_label.return_value = pd.DataFrame()
        self.calculator.get_prob_from_model_x.return_value = np.array([0.3, 0.7])
        self.calculator.get_pred_from_prob_threshold.return_value = np.array([])
        self.calculator.get_report_from_y_prob_pred.return_value = MagicMock()

        self.evaluator.evaluate(args_in, args_al, args_out)

        self.calculator.get_pred_from_prob_threshold.assert_called_once()

    def test_evaluate_writes_json(self):
        args_in = SimpleNamespace(input_csv="test.csv", input_model="model.pt", input_dir="/input")
        args_al = SimpleNamespace(column_label="label", threshold=0.5)
        args_out = SimpleNamespace(output_dir="/output", out_json="out.json", out_md="out.md", out_png="out.png")

        self.reader.read_csv_to_df.return_value = pd.DataFrame()
        self.reader.read_ml_model_from_pt_file.return_value = MagicMock()
        self.calculator.get_output_from_data_label.return_value = np.array([])
        self.calculator.get_input_from_data_label.return_value = pd.DataFrame()
        self.calculator.get_prob_from_model_x.return_value = np.array([])
        self.calculator.get_pred_from_prob_threshold.return_value = np.array([])
        report = MagicMock()
        self.calculator.get_report_from_y_prob_pred.return_value = report

        self.evaluator.evaluate(args_in, args_al, args_out)

        self.writer.write_out_json.assert_called_once()

    def test_evaluate_writes_markdown(self):
        args_in = SimpleNamespace(input_csv="test.csv", input_model="model.pt", input_dir="/input")
        args_al = SimpleNamespace(column_label="label", threshold=0.5)
        args_out = SimpleNamespace(output_dir="/output", out_json="out.json", out_md="out.md", out_png="out.png")

        self.reader.read_csv_to_df.return_value = pd.DataFrame()
        self.reader.read_ml_model_from_pt_file.return_value = MagicMock()
        self.calculator.get_output_from_data_label.return_value = np.array([])
        self.calculator.get_input_from_data_label.return_value = pd.DataFrame()
        self.calculator.get_prob_from_model_x.return_value = np.array([])
        self.calculator.get_pred_from_prob_threshold.return_value = np.array([])
        report = MagicMock()
        self.calculator.get_report_from_y_prob_pred.return_value = report

        self.evaluator.evaluate(args_in, args_al, args_out)

        self.writer.write_out_md.assert_called_once()

    def test_evaluate_creates_plot(self):
        args_in = SimpleNamespace(input_csv="test.csv", input_model="model.pt", input_dir="/input")
        args_al = SimpleNamespace(column_label="label", threshold=0.5)
        args_out = SimpleNamespace(output_dir="/output", out_json="out.json", out_md="out.md", out_png="out.png")

        self.reader.read_csv_to_df.return_value = pd.DataFrame()
        self.reader.read_ml_model_from_pt_file.return_value = MagicMock()
        self.calculator.get_output_from_data_label.return_value = np.array([])
        self.calculator.get_input_from_data_label.return_value = pd.DataFrame()
        self.calculator.get_prob_from_model_x.return_value = np.array([])
        self.calculator.get_pred_from_prob_threshold.return_value = np.array([])
        report = MagicMock()
        self.calculator.get_report_from_y_prob_pred.return_value = report

        self.evaluator.evaluate(args_in, args_al, args_out)

        self.writer.create_plot_of_confusion_matrix.assert_called_once()
