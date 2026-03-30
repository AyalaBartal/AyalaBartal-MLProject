import unittest
from unittest.mock import MagicMock, call, sentinel

from src.specific.lr.evaluate.pe_lr_evaluator import LrPeDataEvaluator
from src.specific.lr.evaluate.pe_lr_evaluate_input_args import LrPeEvaluateInputArgs
from src.specific.lr.evaluate.pe_lr_evaluate_algo_args import LrPeEvaluateAlgoArgs
from src.specific.lr.evaluate.pe_lr_evaluate_output_args import LrPeEvaluateOutputArgs
from src.specific.lr.evaluate.pe_lr_evaluate_report import LrPeEvaluateReport


class TestLrPeDataEvaluator(unittest.TestCase):

    def setUp(self):
        self.reader = MagicMock()
        self.calculator = MagicMock()
        self.writer = MagicMock()

        self.evaluator = LrPeDataEvaluator(
            reader=self.reader,
            calculator=self.calculator,
            writer=self.writer
        )

    def test_init_sets_dependencies(self):
        self.assertIs(self.reader, self.evaluator.reader)
        self.assertIs(self.calculator, self.evaluator.calculator)
        self.assertIs(self.writer, self.evaluator.writer)

    def test_evaluate_calls_all_dependencies_with_expected_args(self):
        args_in = LrPeEvaluateInputArgs("input_dir")
        args_al = LrPeEvaluateAlgoArgs()
        args_out = LrPeEvaluateOutputArgs("output_dir")

        df = sentinel.df
        model = sentinel.model
        y = sentinel.y
        x = sentinel.x
        prob = sentinel.prob
        pred = sentinel.pred
        report = LrPeEvaluateReport(auc=0.91, acc=0.85, cm=[[10, 2], [3, 15]])

        self.reader.read_csv_to_df.return_value = df
        self.reader.read_lr_model_from_joblib_file.return_value = model
        self.calculator.get_output_from_data_label.return_value = y
        self.calculator.get_input_from_data_label.return_value = x
        self.calculator.get_prob_from_model_x.return_value = prob
        self.calculator.get_pred_from_prob_threshold.return_value = pred
        self.calculator.get_report_from_y_prob_pred.return_value = report

        self.evaluator.evaluate(args_in, args_al, args_out)

        self.reader.validate_input.assert_called_once_with(args_in)
        self.reader.validate_output.assert_called_once_with(args_out)
        self.reader.read_csv_to_df.assert_called_once_with(args_in.input_csv)
        self.reader.read_lr_model_from_joblib_file.assert_called_once_with(args_in)

        self.calculator.get_output_from_data_label.assert_called_once_with(args_al, df)
        self.calculator.get_input_from_data_label.assert_called_once_with(args_al, df)
        self.calculator.get_prob_from_model_x.assert_called_once_with(model, x)
        self.calculator.get_pred_from_prob_threshold.assert_called_once_with(prob, args_al.threshold)
        self.calculator.get_report_from_y_prob_pred.assert_called_once_with(y, prob, pred)

        self.writer.write_out_json.assert_called_once_with(args_out.out_json, args_al.threshold, report)
        self.writer.write_out_md.assert_called_once_with(args_out.out_md, args_al.threshold, report)
        self.writer.create_plot_of_confusion_matrix.assert_called_once_with(args_out.out_png, report)

    def test_evaluate_calls_reader_in_expected_order(self):
        args_in = LrPeEvaluateInputArgs("input_dir")
        args_al = LrPeEvaluateAlgoArgs()
        args_out = LrPeEvaluateOutputArgs("output_dir")

        self.reader.read_csv_to_df.return_value = sentinel.df
        self.reader.read_lr_model_from_joblib_file.return_value = sentinel.model
        self.calculator.get_output_from_data_label.return_value = sentinel.y
        self.calculator.get_input_from_data_label.return_value = sentinel.x
        self.calculator.get_prob_from_model_x.return_value = sentinel.prob
        self.calculator.get_pred_from_prob_threshold.return_value = sentinel.pred
        self.calculator.get_report_from_y_prob_pred.return_value = sentinel.report

        self.evaluator.evaluate(args_in, args_al, args_out)

        self.assertEqual(
            self.reader.method_calls,
            [
                call.validate_input(args_in),
                call.validate_output(args_out),
                call.read_csv_to_df(args_in.input_csv),
                call.read_lr_model_from_joblib_file(args_in),
            ]
        )

    def test_evaluate_stops_when_validate_input_raises(self):
        args_in = LrPeEvaluateInputArgs("input_dir")
        args_al = LrPeEvaluateAlgoArgs()
        args_out = LrPeEvaluateOutputArgs("output_dir")

        self.reader.validate_input.side_effect = ValueError("bad input")

        with self.assertRaises(ValueError):
            self.evaluator.evaluate(args_in, args_al, args_out)

        self.reader.validate_output.assert_not_called()
        self.reader.read_csv_to_df.assert_not_called()
        self.reader.read_lr_model_from_joblib_file.assert_not_called()
        self.calculator.get_output_from_data_label.assert_not_called()
        self.writer.write_out_json.assert_not_called()

    def test_evaluate_stops_when_calculator_raises(self):
        args_in = LrPeEvaluateInputArgs("input_dir")
        args_al = LrPeEvaluateAlgoArgs()
        args_out = LrPeEvaluateOutputArgs("output_dir")

        df = sentinel.df
        model = sentinel.model
        y = sentinel.y

        self.reader.read_csv_to_df.return_value = df
        self.reader.read_lr_model_from_joblib_file.return_value = model
        self.calculator.get_output_from_data_label.return_value = y
        self.calculator.get_input_from_data_label.side_effect = RuntimeError("calc failed")

        with self.assertRaises(RuntimeError):
            self.evaluator.evaluate(args_in, args_al, args_out)

        self.calculator.get_prob_from_model_x.assert_not_called()
        self.calculator.get_pred_from_prob_threshold.assert_not_called()
        self.calculator.get_report_from_y_prob_pred.assert_not_called()
        self.writer.write_out_json.assert_not_called()
        self.writer.write_out_md.assert_not_called()
        self.writer.create_plot_of_confusion_matrix.assert_not_called()

    def test_evaluate_uses_threshold_from_algo_args(self):
        args_in = LrPeEvaluateInputArgs("input_dir")
        args_al = LrPeEvaluateAlgoArgs()
        args_out = LrPeEvaluateOutputArgs("output_dir")
        args_al.threshold = 0.75

        report = sentinel.report

        self.reader.read_csv_to_df.return_value = sentinel.df
        self.reader.read_lr_model_from_joblib_file.return_value = sentinel.model
        self.calculator.get_output_from_data_label.return_value = sentinel.y
        self.calculator.get_input_from_data_label.return_value = sentinel.x
        self.calculator.get_prob_from_model_x.return_value = sentinel.prob
        self.calculator.get_pred_from_prob_threshold.return_value = sentinel.pred
        self.calculator.get_report_from_y_prob_pred.return_value = report

        self.evaluator.evaluate(args_in, args_al, args_out)

        self.calculator.get_pred_from_prob_threshold.assert_called_once_with(sentinel.prob, 0.75)
        self.writer.write_out_json.assert_called_once_with(args_out.out_json, 0.75, report)
        self.writer.write_out_md.assert_called_once_with(args_out.out_md, 0.75, report)
