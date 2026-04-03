from src.specific.cbst.evaluate.pe_cbst_evaluate_input_args import CbstPeEvaluateInputArgs
from src.specific.cbst.evaluate.pe_cbst_evaluate_algo_args import CbstPeEvaluateAlgoArgs
from src.specific.cbst.evaluate.pe_cbst_evaluate_output_args import CbstPeEvaluateOutputArgs

from src.specific.cbst.evaluate.pe_cbst_evaluator_reader import CbstPeEvaluatorReader
from src.specific.cbst.evaluate.pe_cbst_evaluator_calculator import CbstPeEvaluatorCalculator
from src.specific.cbst.evaluate.pe_cbst_evaluator_writer import CbstPeEvaluatorWriter


class CbstPeDataEvaluator:
    """Main orchestrator for CatBoost model evaluation.
    
    Coordinates the evaluation workflow including reading data and models,
    calculating metrics, and writing results.
    """

    def __init__(self, reader: CbstPeEvaluatorReader, calculator: CbstPeEvaluatorCalculator, writer: CbstPeEvaluatorWriter):
        """Initialize evaluator with dependencies.
        
        Args:
            reader: Reader for loading models and data.
            calculator: Calculator for computing evaluation metrics.
            writer: Writer for saving evaluation results.
        """
        self.reader = reader
        self.calculator = calculator
        self.writer = writer

    def evaluate(self, args_in: CbstPeEvaluateInputArgs, args_al: CbstPeEvaluateAlgoArgs, args_out: CbstPeEvaluateOutputArgs):
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
        model = self.reader.read_cbst_model_from_joblib_file(args_in)

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
