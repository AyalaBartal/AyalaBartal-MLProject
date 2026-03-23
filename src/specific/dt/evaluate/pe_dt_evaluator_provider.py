from src.common.image.file_io_validator import FileIoValidator
from src.common.plot.ml_io_plot_writer import MlIoPlotWriter
from src.common.plot.matplotlib_plot_renderer import  MatplotlibPlotRenderer
from src.common.plot.matplotlib_plot_exporter import MatplotlibPlotExporter

from src.common.plot.confusion_matrix_spec_factory import ConfusionMatrixPlotSpecFactory
from src.specific.dt.evaluate import DtPeEvaluatorWriter, DtPeEvaluatorFormatter, FileWriter
from src.specific.dt.evaluate.pe_dt_evaluator_calculator import DtPeEvaluatorCalculator
from src.specific.dt.evaluate.pe_dt_evaluator import DtPeDataEvaluator
from src.common.validator.file_validator import FileValidator
from src.specific.dt.evaluate.pe_dt_evaluator_reader import DtPeEvaluatorReader


class DtPeEvaluatorProvider:

    def get_evaluator(self):
        reader = self.get_evaluator_reader()
        calculator = DtPeEvaluatorCalculator()
        writer = self.get_evaluator_writer()
        return DtPeDataEvaluator(reader, calculator, writer)

    def get_evaluator_reader(self):
        dir_validator = FileValidator()
        return DtPeEvaluatorReader(dir_validator)

    def get_evaluator_writer(self):
        formatter = DtPeEvaluatorFormatter()
        file_writer = FileWriter()
        plot_writer = self.get_plot_writer()
        return DtPeEvaluatorWriter(formatter, file_writer, plot_writer)

    def get_plot_writer(self):
        file_validator = FileIoValidator()
        spec_factory = ConfusionMatrixPlotSpecFactory()
        plot_renderer = MatplotlibPlotRenderer()
        plot_exporter = MatplotlibPlotExporter(plot_renderer)
        plot_writer = MlIoPlotWriter(file_validator, spec_factory, plot_exporter)
        return plot_writer
