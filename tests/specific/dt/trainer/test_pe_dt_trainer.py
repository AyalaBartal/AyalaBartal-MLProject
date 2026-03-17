import unittest
import os
import time

from src.specific.dt.trainer.pe_dt_train_algo_args import DtPeTrainAlgoArgs
from src.specific.dt.trainer.pe_dt_train_report_args import DtPeTrainReportArgs
from src.specific.dt.trainer.pe_dt_io_trainer import DtPeIoTrainer
from src.specific.dt.trainer.pe_dt_train_writer import DtPeTrainWriter
from src.specific.dt.trainer.pe_dt_train_output_mapper import DtPeTrainOutputMapper
from src.specific.dt.trainer.pe_dt_train_output_writer import DtPeTrainOutputWriter
from src.specific.dt.trainer.pe_dt_trainer import DtPeDataTrainer
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
        alog_args = DtPeTrainAlgoArgs()
        report_args = DtPeTrainReportArgs(self.input_file, self.out_dir)
        data_trainer = DtPeDataTrainer()
        dt_output_mapper = DtPeTrainOutputMapper()
        output_writer = DtPeTrainOutputWriter()
        dt_writer = DtPeTrainWriter(dt_output_mapper, output_writer)
        io_trainer = DtPeIoTrainer(data_trainer, dt_writer)
        io_trainer.train(alog_args, report_args)
