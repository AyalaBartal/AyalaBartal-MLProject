import unittest
import os
import time

from src.specific.dt.preprocess import DtPeCsvPreprocessMapper, DtPePreprocessorProvider, DtPeDataPreprocessCsvArgs
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
        args = DtPeDataPreprocessCsvArgs(self.input_file, self.output_file)
        mapper = DtPePreprocessorProvider.get_mapper()
        dt_pe_csv_preprocessor = DtPeCsvPreprocessMapper(mapper)
        dt_pe_csv_preprocessor.map(args)


