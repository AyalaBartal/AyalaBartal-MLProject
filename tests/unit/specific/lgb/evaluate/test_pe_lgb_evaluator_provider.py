import unittest

from src.specific.lgb.evaluate.pe_lgb_evaluator_provider import LgbPeEvaluatorProvider
from src.specific.lgb.evaluate.pe_lgb_evaluator import LgbPeDataEvaluator
from src.specific.lgb.evaluate.pe_lgb_evaluator_reader import LgbPeEvaluatorReader
from src.specific.lgb.evaluate.pe_lgb_evaluator_calculator import LgbPeEvaluatorCalculator
from src.specific.lgb.evaluate.pe_lgb_evaluator_formatter import LgbPeEvaluatorFormatter
from src.specific.lgb.evaluate.pe_lgb_evaluator_writer import LgbPeEvaluatorWriter
from src.common.plot.ml_io_plot_writer import MlIoPlotWriter


class TestLgbPeEvaluatorProvider(unittest.TestCase):

    def setUp(self):
        self.provider = LgbPeEvaluatorProvider()

    def test_get_evaluator_returns_lgb_pe_data_evaluator(self):
        actual = self.provider.get_evaluator()

        self.assertIsNotNone(actual)
        self.assertIsInstance(actual, LgbPeDataEvaluator)

    def test_get_evaluator_returns_new_instance_each_time(self):
        first = self.provider.get_evaluator()
        second = self.provider.get_evaluator()

        self.assertIsNot(first, second)

    def test_get_evaluator_builds_expected_reader(self):
        actual = self.provider.get_evaluator()

        self.assertIsNotNone(actual.reader)
        self.assertIsInstance(actual.reader, LgbPeEvaluatorReader)

    def test_get_evaluator_builds_expected_calculator(self):
        actual = self.provider.get_evaluator()

        self.assertIsNotNone(actual.calculator)
        self.assertIsInstance(actual.calculator, LgbPeEvaluatorCalculator)

    def test_get_evaluator_builds_expected_writer(self):
        actual = self.provider.get_evaluator()

        self.assertIsNotNone(actual.writer)
        self.assertIsInstance(actual.writer, LgbPeEvaluatorWriter)

    def test_get_evaluator_writer_has_formatter(self):
        actual = self.provider.get_evaluator()

        self.assertIsNotNone(actual.writer.formatter)
        self.assertIsInstance(actual.writer.formatter, LgbPeEvaluatorFormatter)

    def test_get_evaluator_builds_full_object_graph(self):
        actual = self.provider.get_evaluator()

        # Verify main object
        self.assertIsNotNone(actual)

        # Verify all direct dependencies exist
        self.assertIsNotNone(actual.reader)
        self.assertIsNotNone(actual.calculator)
        self.assertIsNotNone(actual.writer)

        # Verify writer dependencies
        self.assertIsNotNone(actual.writer.formatter)
        self.assertIsNotNone(actual.writer.file_writer)
        self.assertIsNotNone(actual.writer.plot_writer)

    def test_get_evaluator_reader_returns_reader(self):
        actual = self.provider.get_evaluator_reader()

        self.assertIsNotNone(actual)
        self.assertIsInstance(actual, LgbPeEvaluatorReader)

    def test_get_evaluator_writer_returns_writer(self):
        actual = self.provider.get_evaluator_writer()

        self.assertIsNotNone(actual)
        self.assertIsInstance(actual, LgbPeEvaluatorWriter)

    def test_get_plot_writer_returns_plot_writer(self):
        actual = self.provider.get_plot_writer()

        self.assertIsNotNone(actual)
        self.assertIsInstance(actual, MlIoPlotWriter)

    def test_multiple_get_evaluator_calls_return_different_instances(self):
        evaluator1 = self.provider.get_evaluator()
        evaluator2 = self.provider.get_evaluator()
        evaluator3 = self.provider.get_evaluator()

        self.assertIsNot(evaluator1, evaluator2)
        self.assertIsNot(evaluator2, evaluator3)
        self.assertIsNot(evaluator1, evaluator3)


if __name__ == "__main__":
    unittest.main()
