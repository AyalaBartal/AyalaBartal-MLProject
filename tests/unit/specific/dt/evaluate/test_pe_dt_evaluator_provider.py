import unittest

from src.common.validator.file_validator import FileValidator
from src.common.image.file_io_validator import FileIoValidator
from src.specific.dt.evaluate.pe_dt_file_writer import FileWriter
from src.specific.dt.evaluate.pe_dt_evaluator import DtPeDataEvaluator
from src.specific.dt.evaluate.pe_dt_evaluator_provider import DtPeEvaluatorProvider
from src.specific.dt.evaluate.pe_dt_evaluator_reader import DtPeEvaluatorReader
from src.specific.dt.evaluate.pe_dt_evaluator_calculator import DtPeEvaluatorCalculator
from src.specific.dt.evaluate.pe_dt_evaluator_writer import DtPeEvaluatorWriter
from src.specific.dt.evaluate.pe_dt_evaluator_formatter import DtPeEvaluatorFormatter
from src.common.plot.ml_io_plot_writer import MlIoPlotWriter
from src.common.plot.confusion_matrix_spec_factory import ConfusionMatrixPlotSpecFactory
from src.common.plot.matplotlib_plot_renderer import MatplotlibPlotRenderer
from src.common.plot.matplotlib_plot_exporter import MatplotlibPlotExporter


class TestDtPeEvaluatorProvider(unittest.TestCase):

    def setUp(self):
        self.provider = DtPeEvaluatorProvider()

    def test_get_evaluator_returns_dt_pe_data_evaluator(self):
        actual = self.provider.get_evaluator()

        self.assertIsInstance(actual, DtPeDataEvaluator)
        self.assertIsInstance(actual.reader, DtPeEvaluatorReader)
        self.assertIsInstance(actual.calculator, DtPeEvaluatorCalculator)
        self.assertIsInstance(actual.writer, DtPeEvaluatorWriter)

    def test_get_evaluator_reader_returns_reader_with_file_util(self):
        actual = self.provider.get_evaluator_reader()

        self.assertIsInstance(actual, DtPeEvaluatorReader)
        self.assertIsInstance(actual.validator, FileValidator)

    def test_get_evaluator_writer_returns_writer_with_expected_dependencies(self):
        actual = self.provider.get_evaluator_writer()

        self.assertIsInstance(actual, DtPeEvaluatorWriter)
        self.assertIsInstance(actual.formatter, DtPeEvaluatorFormatter)
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
