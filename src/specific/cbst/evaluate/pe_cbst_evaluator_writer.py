from src.common.plot import MlIoPlotWriter
from src.specific.cbst.evaluate.pe_cbst_evaluator_formatter import CbstPeEvaluatorFormatter
from src.specific.cbst.evaluate.pe_cbst_file_writer import FileWriter


class CbstPeEvaluatorWriter:
    """Writer for CatBoost evaluation results.
    
    Orchestrates writing evaluation metrics, summaries, and confusion matrix plots
    to various output formats (JSON, Markdown, PNG).
    """

    def __init__(self, formatter: CbstPeEvaluatorFormatter, file_writer: FileWriter, plot_writer: MlIoPlotWriter):
        """Initialize writer with dependencies.
        
        Args:
            formatter: Formatter for converting reports to output formats.
            file_writer: FileWriter for writing to files.
            plot_writer: Plot writer for creating confusion matrix visualizations.
        """
        self.formatter = formatter
        self.plot_writer = plot_writer
        self.file_writer = file_writer

    def write_out_json(self, out_json: str, threshold: float, report):
        """Write evaluation metrics to JSON file.
        
        Args:
            out_json: Path to output JSON file.
            threshold: Classification threshold used.
            report: CbstPeEvaluateReport instance.
        """
        json_data = self.formatter.get_json_from_report_threshold(report, threshold)
        self.file_writer.write_out_json(out_json, json_data)

    def write_out_md(self, out_file: str, threshold: float, report):
        """Write evaluation summary to Markdown file.
        
        Args:
            out_file: Path to output Markdown file.
            threshold: Classification threshold used.
            report: CbstPeEvaluateReport instance.
        """
        out_text = self.formatter.get_md_from_report_threshold(report, threshold)
        self.file_writer.write_out_md(out_file, out_text)

    def create_plot_of_confusion_matrix(self, out_png: str, report):
        """Create and save confusion matrix plot.
        
        Args:
            out_png: Path to output PNG file.
            report: CbstPeEvaluateReport instance.
        """
        self.plot_writer.create_plot(out_png, report.cm)
