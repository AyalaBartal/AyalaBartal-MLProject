from src.specific.lgb.evaluate.pe_lgb_evaluate_input_args import LgbPeEvaluateInputArgs
from src.specific.lgb.evaluate.pe_lgb_evaluate_algo_args import LgbPeEvaluateAlgoArgs
from src.specific.lgb.evaluate.pe_lgb_evaluate_output_args import LgbPeEvaluateOutputArgs

from src.specific.lgb.evaluate.pe_lgb_evaluator_reader import LgbPeEvaluatorReader
from src.specific.lgb.evaluate.pe_lgb_evaluator_calculator import LgbPeEvaluatorCalculator
from src.specific.lgb.evaluate.pe_lgb_evaluator_writer import LgbPeEvaluatorWriter


class LgbPeDataEvaluator:
    """Main orchestrator for LightGBM model evaluation.
    
    Coordinates the evaluation workflow including reading data and models,
    calculating metrics, and writing results.
    """

    def __init__(self, reader: LgbPeEvaluatorReader, calculator: LgbPeEvaluatorCalculator, writer: LgbPeEvaluatorWriter):
        """Initialize evaluator with dependencies.
        
        Args:
            reader: Reader for loading models and data.
            calculator: Calculator for computing evaluation metrics.
            writer: Writer for saving evaluation results.
        """
        self.reader = reader
        self.calculator = calculator
        self.writer = writer

    def evaluate(self, args_in: LgbPeEvaluateInputArgs, args_al: LgbPeEvaluateAlgoArgs, args_out: LgbPeEvaluateOutputArgs):
        """Execute complete evaluation workflow.
        
        Steps:
        1. Read and validate input data and model
        2. Calculate metrics (AUC, accuracy, confusion matrix)
        3. Write results to JSON, Markdown, and PNG files
        
        Args:
            args_in: Input arguments (paths to data and model).
            args_al: Algorithm arguments (threshold, label column).
            args_out: Output arguments (output paths).
        """
        # 1) Read data
        self.reader.validate_input(args_in)
        self.reader.validate_output(args_out)
        df = self.reader.read_csv_to_df(args_in.input_csv)
        model = self.reader.read_lgb_model_from_joblib_file(args_in)

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
