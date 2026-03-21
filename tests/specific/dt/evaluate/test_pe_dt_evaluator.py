import unittest
import os
import time

from src.common.image.file_io_validator import FileIoValidator
from src.common.plot.ml_io_plot_writer import MlIoPlotWriter
from src.common.plot.matplotlib_plot_renderer import  MatplotlibPlotRenderer
from src.common.plot.matplotlib_plot_exporter import MatplotlibPlotExporter

from src.common.plot.confusion_matrix_spec_factory import ConfusionMatrixPlotSpecFactory
from src.specific.dt.evaluate import DtPeEvaluatorWriter, DtPeEvaluatorFormatter, FileWriter
from src.specific.dt.evaluate.pe_dt_evaluator_calculator import DtPeEvaluatorCalculator
from src.specific.dt.evaluate.pe_dt_evaluator import DtPeDataEvaluator
from src.specific.dt.evaluate.file_util import FileUtil
from src.specific.dt.evaluate.pe_dt_evaluate_input_args import DtPeEvaluateInputArgs
from src.specific.dt.evaluate.pe_dt_evaluate_algo_args import DtPeEvaluateAlgoArgs
from src.specific.dt.evaluate.pe_dt_evaluate_output_args import DtPeEvaluateOutputArgs
from src.specific.dt.evaluate.pe_dt_evaluator_reader import DtPeEvaluatorReader

from tests.utils import PathsProvider


class TestDtPeDataEvaluator(unittest.TestCase):

    def setUp(self):
        self.start_time = time.perf_counter()  # Precise timing [15]

        data_dir = PathsProvider.get_test_data_dir()
        print('data_dir={}'.format(data_dir))
        self.input_dir = os.path.join(data_dir, 'specific', 'dt', 'evaluate', 'input')
        self.output_dir = os.path.join(data_dir, 'specific', 'dt', 'evaluate', 'output')
        print('input_dir={}, output_dir={}'.format(self.input_dir, self.output_dir))

    def tearDown(self):
        duration = time.perf_counter() - self.start_time
        print(f"\n{self.id()} took {duration:.4f} seconds")

    def test_evaluate_dt_model_success(self):
        input_args = DtPeEvaluateInputArgs(self.input_dir)
        algo_args = DtPeEvaluateAlgoArgs()
        output_args = DtPeEvaluateOutputArgs(self.output_dir)

        dir_validator = FileUtil()
        reader = DtPeEvaluatorReader(dir_validator)

        calculator = DtPeEvaluatorCalculator()

        formatter = DtPeEvaluatorFormatter()
        file_writer = FileWriter()
        plot_writer = self.get_plot_writer()
        writer = DtPeEvaluatorWriter(formatter, file_writer, plot_writer)

        evaluator = DtPeDataEvaluator(reader, calculator, writer)
        evaluator.evaluate(input_args, algo_args, output_args)

    def get_plot_writer(self):
        file_validator = FileIoValidator()
        spec_factory = ConfusionMatrixPlotSpecFactory()
        plot_renderer = MatplotlibPlotRenderer()
        plot_exporter = MatplotlibPlotExporter(plot_renderer)
        plot_writer = MlIoPlotWriter(file_validator, spec_factory, plot_exporter)
        return plot_writer
