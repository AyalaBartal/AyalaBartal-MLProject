from src.common.image.file_io_validator import FileIoValidator
from src.common.plot.ml_io_plot_writer import MlIoPlotWriter
from src.common.plot.matplotlib_plot_renderer import MatplotlibPlotRenderer
from src.common.plot.matplotlib_plot_exporter import MatplotlibPlotExporter

from src.common.plot.confusion_matrix_spec_factory import ConfusionMatrixPlotSpecFactory
from src.specific.rf.evaluate.pe_rf_evaluator_writer import RfPeEvaluatorWriter
from src.specific.rf.evaluate.pe_rf_evaluator_formatter import RfPeEvaluatorFormatter
from src.specific.rf.evaluate.pe_rf_file_writer import FileWriter
from src.specific.rf.evaluate.pe_rf_evaluator_calculator import RfPeEvaluatorCalculator
from src.specific.rf.evaluate.pe_rf_evaluator import RfPeDataEvaluator
from src.common.validator.file_validator import FileValidator
from src.specific.rf.evaluate.pe_rf_evaluator_reader import RfPeEvaluatorReader


class RfPeEvaluatorProvider:

    def get_evaluator(self):
        reader = self.get_evaluator_reader()
        calculator = RfPeEvaluatorCalculator()
        writer = self.get_evaluator_writer()
        return RfPeDataEvaluator(reader, calculator, writer)

    def get_evaluator_reader(self):
        dir_validator = FileValidator()
        return RfPeEvaluatorReader(dir_validator)

    def get_evaluator_writer(self):
        formatter = RfPeEvaluatorFormatter()
        file_writer = FileWriter()
        plot_writer = self.get_plot_writer()
        return RfPeEvaluatorWriter(formatter, file_writer, plot_writer)

    def get_plot_writer(self):
        file_validator = FileIoValidator()
        spec_factory = ConfusionMatrixPlotSpecFactory()
        plot_renderer = MatplotlibPlotRenderer()
        plot_exporter = MatplotlibPlotExporter(plot_renderer)
        plot_writer = MlIoPlotWriter(file_validator, spec_factory, plot_exporter)
        return plot_writer
