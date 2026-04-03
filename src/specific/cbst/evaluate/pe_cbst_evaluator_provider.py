from src.common.image.file_io_validator import FileIoValidator
from src.common.plot.ml_io_plot_writer import MlIoPlotWriter
from src.common.plot.matplotlib_plot_renderer import MatplotlibPlotRenderer
from src.common.plot.matplotlib_plot_exporter import MatplotlibPlotExporter

from src.common.plot.confusion_matrix_spec_factory import ConfusionMatrixPlotSpecFactory
from src.specific.cbst.evaluate.pe_cbst_evaluator_writer import CbstPeEvaluatorWriter
from src.specific.cbst.evaluate.pe_cbst_evaluator_formatter import CbstPeEvaluatorFormatter
from src.specific.cbst.evaluate.pe_cbst_file_writer import FileWriter
from src.specific.cbst.evaluate.pe_cbst_evaluator_calculator import CbstPeEvaluatorCalculator
from src.specific.cbst.evaluate.pe_cbst_evaluator import CbstPeDataEvaluator
from src.common.validator.file_validator import FileValidator
from src.specific.cbst.evaluate.pe_cbst_evaluator_reader import CbstPeEvaluatorReader


class CbstPeEvaluatorProvider:
    """Dependency injection provider for CatBoost evaluator.
    
    Factory for creating fully configured CbstPeDataEvaluator instance
    with all necessary dependencies injected.
    """

    def get_evaluator(self) -> CbstPeDataEvaluator:
        """Create and return configured evaluator instance.
        
        Returns:
            CbstPeDataEvaluator with all dependencies injected.
        """
        reader = self.get_evaluator_reader()
        calculator = CbstPeEvaluatorCalculator()
        writer = self.get_evaluator_writer()
        return CbstPeDataEvaluator(reader, calculator, writer)

    def get_evaluator_reader(self) -> CbstPeEvaluatorReader:
        """Create and return configured reader instance.
        
        Returns:
            CbstPeEvaluatorReader with file validator.
        """
        dir_validator = FileValidator()
        return CbstPeEvaluatorReader(dir_validator)

    def get_evaluator_writer(self) -> CbstPeEvaluatorWriter:
        """Create and return configured writer instance.
        
        Returns:
            CbstPeEvaluatorWriter with formatter, file writer, and plot writer.
        """
        formatter = CbstPeEvaluatorFormatter()
        file_writer = FileWriter()
        plot_writer = self.get_plot_writer()
        return CbstPeEvaluatorWriter(formatter, file_writer, plot_writer)

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
