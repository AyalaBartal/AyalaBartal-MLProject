from src.common.image.file_io_validator import FileIoValidator
from src.common.plot.ml_io_plot_writer import MlIoPlotWriter
from src.common.plot.matplotlib_plot_renderer import MatplotlibPlotRenderer
from src.common.plot.matplotlib_plot_exporter import MatplotlibPlotExporter

from src.common.plot.confusion_matrix_spec_factory import ConfusionMatrixPlotSpecFactory
from src.specific.lr.evaluate.pe_lr_evaluator_writer import LrPeEvaluatorWriter
from src.specific.lr.evaluate.pe_lr_evaluator_formatter import LrPeEvaluatorFormatter
from src.specific.lr.evaluate.pe_lr_file_writer import FileWriter
from src.specific.lr.evaluate.pe_lr_evaluator_calculator import LrPeEvaluatorCalculator
from src.specific.lr.evaluate.pe_lr_evaluator import LrPeDataEvaluator
from src.common.validator.file_validator import FileValidator
from src.specific.lr.evaluate.pe_lr_evaluator_reader import LrPeEvaluatorReader


class LrPeEvaluatorProvider:

    def get_evaluator(self):
        reader = self.get_evaluator_reader()
        calculator = LrPeEvaluatorCalculator()
        writer = self.get_evaluator_writer()
        return LrPeDataEvaluator(reader, calculator, writer)

    def get_evaluator_reader(self):
        dir_validator = FileValidator()
        return LrPeEvaluatorReader(dir_validator)

    def get_evaluator_writer(self):
        formatter = LrPeEvaluatorFormatter()
        file_writer = FileWriter()
        plot_writer = self.get_plot_writer()
        return LrPeEvaluatorWriter(formatter, file_writer, plot_writer)

    def get_plot_writer(self):
        file_validator = FileIoValidator()
        spec_factory = ConfusionMatrixPlotSpecFactory()
        plot_renderer = MatplotlibPlotRenderer()
        plot_exporter = MatplotlibPlotExporter(plot_renderer)
        plot_writer = MlIoPlotWriter(file_validator, spec_factory, plot_exporter)
        return plot_writer
