import unittest

from src.common.validator.file_validator import FileValidator
from src.common.image.file_io_validator import FileIoValidator
from src.specific.rf.evaluate.pe_rf_file_writer import FileWriter
from src.specific.rf.evaluate.pe_rf_evaluator import RfPeDataEvaluator
from src.specific.rf.evaluate.pe_rf_evaluator_provider import RfPeEvaluatorProvider
from src.specific.rf.evaluate.pe_rf_evaluator_reader import RfPeEvaluatorReader
from src.specific.rf.evaluate.pe_rf_evaluator_calculator import RfPeEvaluatorCalculator
from src.specific.rf.evaluate.pe_rf_evaluator_writer import RfPeEvaluatorWriter
from src.specific.rf.evaluate.pe_rf_evaluator_formatter import RfPeEvaluatorFormatter
from src.common.plot.ml_io_plot_writer import MlIoPlotWriter
from src.common.plot.confusion_matrix_spec_factory import ConfusionMatrixPlotSpecFactory
from src.common.plot.matplotlib_plot_renderer import MatplotlibPlotRenderer
from src.common.plot.matplotlib_plot_exporter import MatplotlibPlotExporter


class TestRfPeEvaluatorProvider(unittest.TestCase):

    def setUp(self):
        self.provider = RfPeEvaluatorProvider()

    def test_get_evaluator_returns_rf_pe_data_evaluator(self):
        actual = self.provider.get_evaluator()

        self.assertIsInstance(actual, RfPeDataEvaluator)
        self.assertIsInstance(actual.reader, RfPeEvaluatorReader)
        self.assertIsInstance(actual.calculator, RfPeEvaluatorCalculator)
        self.assertIsInstance(actual.writer, RfPeEvaluatorWriter)

    def test_get_evaluator_reader_returns_reader_with_file_util(self):
        actual = self.provider.get_evaluator_reader()

        self.assertIsInstance(actual, RfPeEvaluatorReader)
        self.assertIsInstance(actual.validator, FileValidator)

    def test_get_evaluator_writer_returns_writer_with_expected_dependencies(self):
        actual = self.provider.get_evaluator_writer()

        self.assertIsInstance(actual, RfPeEvaluatorWriter)
        self.assertIsInstance(actual.formatter, RfPeEvaluatorFormatter)
        self.assertIsInstance(actual.file_writer, FileWriter)
        self.assertIsInstance(actual.plot_writer, MlIoPlotWriter)

    def test_get_plot_writer_returns_ml_io_plot_writer_with_expected_dependencies(self):
        actual = self.provider.get_plot_writer()

        self.assertIsInstance(actual, MlIoPlotWriter)
        self.assertIsInstance(actual.file_validator, FileIoValidator)
        self.assertIsInstance(actual.spec_factory, ConfusionMatrixPlotSpecFactory)
        self.assertIsInstance(actual.plot_exporter, MatplotlibPlotExporter)

    def test_get_plot_writer_builds_exporter_with_matplotlib_plot_renderer(self):
        actual = self.provider.get_plot_writer()

        self.assertIsInstance(actual.plot_exporter.renderer, MatplotlibPlotRenderer)
