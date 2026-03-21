import json

from src.common.image.file_io_validator import FileIoValidator
from src.common.plot import MlIoPlotWriter, MatplotlibPlotExporter
from src.common.plot.confusion_matrix_spec_factory import ConfusionMatrixPlotSpecFactory
from src.common.plot.matplotlib_plot_renderer import MatplotlibPlotRenderer
from src.specific.dt.evaluate import DtPeEvaluateInputArgs, DtPeEvaluateOutputArgs, DtPeEvaluateAlgoArgs, \
    DtPeEvaluatorWriter
from src.specific.dt.evaluate.pe_dt_evaluator_reader import DtPeEvaluatorReader
from src.specific.dt.evaluate.pe_dt_evaluator_calculator import DtPeEvaluatorCalculator


class DtPeDataEvaluator:

    def __init__(self, reader: DtPeEvaluatorReader, calculator: DtPeEvaluatorCalculator, writer: DtPeEvaluatorWriter):
        self.reader = reader
        self.calculator = calculator
        self.writer = writer

    def evaluate(self, args_in: DtPeEvaluateInputArgs, args_al: DtPeEvaluateAlgoArgs, args_out: DtPeEvaluateOutputArgs):

        print("Start DtPeDataEvaluator")
        self.reader.validate_input(args_in)
        self.reader.validate_output(args_out)

        df = self.reader.read_csv_to_df(args_in.input_csv)
        model = self.reader.read_dt_model_from_joblib_file(args_in)

        y = self.calculator.get_output_from_data_label(args_al, df)
        x = self.calculator.get_input_from_data_label(args_al, df)
        proba = self.calculator.get_prob_from_model_x(model, x)
        y_pred = self.calculator.get_pred_from_prob_threshold(proba, args_al.threshold)
        report = self.calculator.get_report_from_y_prob_pred(y, proba, y_pred)

        self.writer.write_out_json(args_out.out_json, args_al.threshold, report)
        self.writer.write_out_md(args_out.out_md, args_al.threshold, report)
        self.writer.create_plot_of_confusion_matrix(args_out.out_png, report)

        print("End DtPeDataEvaluator")





