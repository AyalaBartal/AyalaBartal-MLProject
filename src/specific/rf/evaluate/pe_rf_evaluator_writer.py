from src.common.plot import MlIoPlotWriter
from src.specific.rf.evaluate.pe_rf_evaluator_formatter import RfPeEvaluatorFormatter
from src.specific.rf.evaluate.pe_rf_file_writer import FileWriter


class RfPeEvaluatorWriter:

    def __init__(self, formatter: RfPeEvaluatorFormatter, file_writer: FileWriter, plot_writer: MlIoPlotWriter):
        self.formatter = formatter
        self.plot_writer = plot_writer
        self.file_writer = file_writer

    def write_out_json(self, out_json, threshold, report):
        json_data = self.formatter.get_json_from_report_threshold(report, threshold)
        self.file_writer.write_out_json(out_json, json_data)

    def write_out_md(self, out_file, threshold, report):
        out_text = self.formatter.get_md_from_report_threshold(report, threshold)
        self.file_writer.write_out_md(out_file, out_text)

    def create_plot_of_confusion_matrix(self, out_png, report):
        self.plot_writer.create_plot(out_png, report.cm)
