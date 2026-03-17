import unittest
import os
import time

from src.specific.dt.trainer.pe_dt_trainer_provider import DtPeTrainerProvider
from src.specific.dt.trainer.pe_dt_train_algo_args import DtPeTrainAlgoArgs
from src.specific.dt.trainer.pe_dt_train_report_args import DtPeTrainReportArgs
from tests.utils import PathsProvider


class TestPeDtPreprocessor(unittest.TestCase):

    def setUp(self):
        self.start_time = time.perf_counter()  # Precise timing [15]

        data_dir = PathsProvider.get_test_data_dir()
        print('data_dir={}'.format(data_dir))

        self.input_file = os.path.join(data_dir, 'specific', 'dt', 'train', 'input', 'dt_input_data.csv')
        self.out_dir = os.path.join(data_dir, 'specific', 'dt', 'train', 'output')
        print('input_file={}, out_dir={}'.format(self.input_file, self.out_dir))

    def tearDown(self):
        duration = time.perf_counter() - self.start_time
        print(f"\n{self.id()} took {duration:.4f} seconds")

    def test_validate_header_of_input_csv_success(self):
        algo_args = DtPeTrainAlgoArgs()
        report_args = DtPeTrainReportArgs(self.input_file, self.out_dir)
        io_trainer = DtPeTrainerProvider.get_io_trainer()
        io_trainer.train(algo_args, report_args)
