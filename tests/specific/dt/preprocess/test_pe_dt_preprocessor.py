import unittest
import os
import time

from src.specific.dt.preprocess import DtPeCsvPreprocessMapper
from src.specific.dt.preprocess.pe_dt_data_transformer import DtPeDataTransformer
from src.specific.dt.preprocess.pe_dt_data_converter import DtPeDataConverter
from src.specific.dt.preprocess.pe_dt_preprocess_args import DtPeDataPreprocessArgs
from tests.utils import PathsProvider


class TestPeDtPreprocessor(unittest.TestCase):

    def setUp(self):
        self.start_time = time.perf_counter()  # Precise timing [15]

        data_dir = PathsProvider.get_test_data_dir()
        print('data_dir={}'.format(data_dir))

        self.input_file = os.path.join(data_dir, 'specific', 'dt', 'preprocess', 'malware_input_data.csv')
        self.output_file = os.path.join(data_dir, 'specific', 'dt', 'preprocess', 'malware_output_data.csv')
        print('input_file={}, output_file={}'.format(self.input_file, self.output_file))

    def tearDown(self):
        duration = time.perf_counter() - self.start_time
        print(f"\n{self.id()} took {duration:.4f} seconds")

    def test_pre_process_input_malware_csv_for_dt_success(self):
        converter = DtPeDataConverter()
        dt_pe_data_transformer = DtPeDataTransformer(converter)
        dt_pe_data_preprocess_args = DtPeDataPreprocessArgs(self.input_file, self.output_file)
        dt_pe_data_preprocessor = DtPeCsvPreprocessMapper(dt_pe_data_transformer)
        dt_pe_data_preprocessor.map(dt_pe_data_preprocess_args)


