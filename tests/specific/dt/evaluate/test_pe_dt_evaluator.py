import unittest
import os
import time

from src.specific.dt.evaluate import DtPeDataEvaluator
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

        evaluator = DtPeDataEvaluator(reader)
        evaluator.evaluate(input_args, algo_args, output_args)
