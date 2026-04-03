from src.common.image.file_io_validator import FileIoValidator
from src.common.plot.ml_io_plot_writer import MlIoPlotWriter
from src.common.plot.matplotlib_plot_renderer import MatplotlibPlotRenderer
from src.common.plot.matplotlib_plot_exporter import MatplotlibPlotExporter

from src.common.plot.confusion_matrix_spec_factory import ConfusionMatrixPlotSpecFactory
from src.specific.lgb.evaluate.pe_lgb_evaluator_writer import LgbPeEvaluatorWriter
from src.specific.lgb.evaluate.pe_lgb_evaluator_formatter import LgbPeEvaluatorFormatter
from src.specific.lgb.evaluate.pe_lgb_file_writer import FileWriter
from src.specific.lgb.evaluate.pe_lgb_evaluator_calculator import LgbPeEvaluatorCalculator
from src.specific.lgb.evaluate.pe_lgb_evaluator import LgbPeDataEvaluator
from src.common.validator.file_validator import FileValidator
from src.specific.lgb.evaluate.pe_lgb_evaluator_reader import LgbPeEvaluatorReader


class LgbPeEvaluatorProvider:
    """Dependency injection provider for LightGBM evaluator.
    
    Factory for creating fully configured LgbPeDataEvaluator instance
    with all necessary dependencies injected.
    """

    def get_evaluator(self) -> LgbPeDataEvaluator:
        """Create and return configured evaluator instance.
        
        Returns:
            LgbPeDataEvaluator with all dependencies injected.
        """
        reader = self.get_evaluator_reader()
        calculator = LgbPeEvaluatorCalculator()
        writer = self.get_evaluator_writer()
        return LgbPeDataEvaluator(reader, calculator, writer)

    def get_evaluator_reader(self) -> LgbPeEvaluatorReader:
        """Create and return configured reader instance.
        
        Returns:
            LgbPeEvaluatorReader with file validator.
        """
        dir_validator = FileValidator()
        return LgbPeEvaluatorReader(dir_validator)

    def get_evaluator_writer(self) -> LgbPeEvaluatorWriter:
        """Create and return configured writer instance.
        
        Returns:
            LgbPeEvaluatorWriter with formatter, file writer, and plot writer.
        """
        formatter = LgbPeEvaluatorFormatter()
        file_writer = FileWriter()
        plot_writer = self.get_plot_writer()
        return LgbPeEvaluatorWriter(formatter, file_writer, plot_writer)

    def get_plot_writer(self) -> MlIoPlotWriter:
        """Create and return configured plot writer instance.
        
        Returns:
            MlIoPlotWriter for creating confusion matrix visualizations.
        """
        file_validator = FileIoValidator()
        spec_factory = ConfusionMatrixPlotSpecFactory()
        plot_renderer = MatplotlibPlotRenderer()
        plot_exporter = MatplotlibPlotExporter(plot_renderer)
        plot_writer = MlIoPlotWriter(file_validator, spec_factory, plot_exporter)
        return plot_writer
