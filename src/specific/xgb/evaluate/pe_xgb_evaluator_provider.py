from src.common.image.file_io_validator import FileIoValidator
from src.common.plot.ml_io_plot_writer import MlIoPlotWriter
from src.common.plot.matplotlib_plot_renderer import MatplotlibPlotRenderer
from src.common.plot.matplotlib_plot_exporter import MatplotlibPlotExporter

from src.common.plot.confusion_matrix_spec_factory import ConfusionMatrixPlotSpecFactory
from src.specific.xgb.evaluate.pe_xgb_evaluator_writer import XgbPeEvaluatorWriter
from src.specific.xgb.evaluate.pe_xgb_evaluator_formatter import XgbPeEvaluatorFormatter
from src.specific.xgb.evaluate.pe_xgb_file_writer import FileWriter
from src.specific.xgb.evaluate.pe_xgb_evaluator_calculator import XgbPeEvaluatorCalculator
from src.specific.xgb.evaluate.pe_xgb_evaluator import XgbPeDataEvaluator
from src.common.validator.file_validator import FileValidator
from src.specific.xgb.evaluate.pe_xgb_evaluator_reader import XgbPeEvaluatorReader


class XgbPeEvaluatorProvider:
    """Dependency injection provider for XGBoost evaluator.
    
    Factory for creating fully configured XgbPeDataEvaluator instance
    with all necessary dependencies injected.
    """

    def get_evaluator(self) -> XgbPeDataEvaluator:
        """Create and return configured evaluator instance.
        
        Returns:
            XgbPeDataEvaluator with all dependencies injected.
        """
        reader = self.get_evaluator_reader()
        calculator = XgbPeEvaluatorCalculator()
        writer = self.get_evaluator_writer()
        return XgbPeDataEvaluator(reader, calculator, writer)

    def get_evaluator_reader(self) -> XgbPeEvaluatorReader:
        """Create and return configured reader instance.
        
        Returns:
            XgbPeEvaluatorReader with file validator.
        """
        dir_validator = FileValidator()
        return XgbPeEvaluatorReader(dir_validator)

    def get_evaluator_writer(self) -> XgbPeEvaluatorWriter:
        """Create and return configured writer instance.
        
        Returns:
            XgbPeEvaluatorWriter with formatter, file writer, and plot writer.
        """
        formatter = XgbPeEvaluatorFormatter()
        file_writer = FileWriter()
        plot_writer = self.get_plot_writer()
        return XgbPeEvaluatorWriter(formatter, file_writer, plot_writer)

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
