import unittest

from src.specific.ml.evaluate.pe_ml_evaluator_provider import MlPeEvaluatorProvider
from src.specific.ml.evaluate.pe_ml_evaluator_reader import MlPeEvaluatorReader
from src.specific.ml.evaluate.pe_ml_evaluator_calculator import MlPeEvaluatorCalculator
from src.specific.ml.evaluate.pe_ml_evaluator_writer import MlPeEvaluatorWriter
from src.specific.ml.evaluate.pe_ml_evaluator import MlPeDataEvaluator
from src.common.plot import MlIoPlotWriter


class TestMlPeEvaluatorProvider(unittest.TestCase):

    def setUp(self):
        self.provider = MlPeEvaluatorProvider()

    def test_get_evaluator_returns_data_evaluator(self):
        evaluator = self.provider.get_evaluator()

        self.assertIsInstance(evaluator, MlPeDataEvaluator)

    def test_get_evaluator_reader_returns_evaluator_reader(self):
        reader = self.provider.get_evaluator_reader()

        self.assertIsInstance(reader, MlPeEvaluatorReader)

    def test_get_evaluator_writer_returns_evaluator_writer(self):
        writer = self.provider.get_evaluator_writer()

        self.assertIsInstance(writer, MlPeEvaluatorWriter)

    def test_get_plot_writer_returns_ml_io_plot_writer(self):
        plot_writer = self.provider.get_plot_writer()

        self.assertIsInstance(plot_writer, MlIoPlotWriter)

    def test_get_evaluator_has_reader(self):
        evaluator = self.provider.get_evaluator()

        self.assertIsInstance(evaluator.reader, MlPeEvaluatorReader)

    def test_get_evaluator_has_calculator(self):
        evaluator = self.provider.get_evaluator()

        self.assertIsInstance(evaluator.calculator, MlPeEvaluatorCalculator)

    def test_get_evaluator_has_writer(self):
        evaluator = self.provider.get_evaluator()

        self.assertIsInstance(evaluator.writer, MlPeEvaluatorWriter)

    def test_get_evaluator_writer_has_formatter(self):
        writer = self.provider.get_evaluator_writer()

        self.assertIsNotNone(writer.formatter)

    def test_get_evaluator_writer_has_file_writer(self):
        writer = self.provider.get_evaluator_writer()

        self.assertIsNotNone(writer.file_writer)

    def test_get_evaluator_writer_has_plot_writer(self):
        writer = self.provider.get_evaluator_writer()

        self.assertIsNotNone(writer.plot_writer)
