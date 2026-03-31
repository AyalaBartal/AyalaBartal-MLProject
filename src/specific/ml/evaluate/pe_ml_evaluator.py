from src.specific.ml.evaluate.pe_ml_evaluate_input_args import MlPeEvaluateInputArgs
from src.specific.ml.evaluate.pe_ml_evaluate_algo_args import MlPeEvaluateAlgoArgs
from src.specific.ml.evaluate.pe_ml_evaluate_output_args import MlPeEvaluateOutputArgs

from src.specific.ml.evaluate.pe_ml_evaluator_reader import MlPeEvaluatorReader
from src.specific.ml.evaluate.pe_ml_evaluator_calculator import MlPeEvaluatorCalculator
from src.specific.ml.evaluate.pe_ml_evaluator_writer import MlPeEvaluatorWriter


class MlPeDataEvaluator:

    def __init__(self, reader: MlPeEvaluatorReader, calculator: MlPeEvaluatorCalculator, writer: MlPeEvaluatorWriter):
        self.reader = reader
        self.calculator = calculator
        self.writer = writer

    def evaluate(self, args_in: MlPeEvaluateInputArgs, args_al: MlPeEvaluateAlgoArgs, args_out: MlPeEvaluateOutputArgs):
        # 1) Read data
        self.reader.validate_input(args_in)
        self.reader.validate_output(args_out)
        df = self.reader.read_csv_to_df(args_in.input_csv)
        model = self.reader.read_ml_model_from_pt_file(args_in)

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
