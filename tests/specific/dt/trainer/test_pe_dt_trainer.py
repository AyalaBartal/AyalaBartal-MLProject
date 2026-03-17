import unittest
import os
import time

from src.specific.dt.trainer import DtPeIoTrainer, DtPeTrainWriter, DtPeTrainOutputMapper, DtPeTrainOutputWriter
from src.specific.dt.trainer.pe_dt_train_args import DtPeTrainArgs
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
        args = DtPeTrainArgs(self.input_file, self.out_dir)
        data_trainer = DtPeDataTrainer()
        dt_output_mapper = DtPeTrainOutputMapper()
        output_writer = DtPeTrainOutputWriter()
        dt_writer = DtPeTrainWriter(dt_output_mapper, output_writer)
        io_trainer = DtPeIoTrainer(data_trainer, dt_writer)
        io_trainer.train(args)
