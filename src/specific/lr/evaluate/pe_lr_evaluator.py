from src.specific.lr.evaluate.pe_lr_evaluate_input_args import LrPeEvaluateInputArgs
from src.specific.lr.evaluate.pe_lr_evaluate_algo_args import LrPeEvaluateAlgoArgs
from src.specific.lr.evaluate.pe_lr_evaluate_output_args import LrPeEvaluateOutputArgs

from src.specific.lr.evaluate.pe_lr_evaluator_reader import LrPeEvaluatorReader
from src.specific.lr.evaluate.pe_lr_evaluator_calculator import LrPeEvaluatorCalculator
from src.specific.lr.evaluate.pe_lr_evaluator_writer import LrPeEvaluatorWriter


class LrPeDataEvaluator:

    def __init__(self, reader: LrPeEvaluatorReader, calculator: LrPeEvaluatorCalculator, writer: LrPeEvaluatorWriter):
        self.reader = reader
        self.calculator = calculator
        self.writer = writer

    def evaluate(self, args_in: LrPeEvaluateInputArgs, args_al: LrPeEvaluateAlgoArgs, args_out: LrPeEvaluateOutputArgs):
        # 1) Read data
        self.reader.validate_input(args_in)
        self.reader.validate_output(args_out)
        df = self.reader.read_csv_to_df(args_in.input_csv)
        model = self.reader.read_lr_model_from_joblib_file(args_in)

        # 2) Calculate report from data
        y = self.calculator.get_output_from_data_label(args_al, df)
        x = self.calculator.get_input_from_data_label(args_al, df)
        prob = self.calculator.get_prob_from_model_x(model, x)
        pred = self.calculator.get_pred_from_prob_threshold(prob, args_al.threshold)
        report = self.calculator.get_report_from_y_prob_pred(y, prob, pred)

        # 3) Write report to files
        self.writer.write_out_json(args_out.out_json, args_al.threshold, report)
        self.writer.write_out_md(args_out.out_md, args_al.threshold, report)
        self.writer.create_plot_of_confusion_matrix(args_out.out_png, report)
