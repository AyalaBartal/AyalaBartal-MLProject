import unittest
import os
import time
from src.common.csv import CsvBrazilianProvider
from src.common.csv import CsvIoValidate
from tests.utils import PathsProvider


class TestPePreprocessor(unittest.TestCase):

    def setUp(self):
        self.start_time = time.perf_counter()  # Precise timing [15]

        data_dir = PathsProvider.get_test_data_dir()
        print('data_dir={}'.format(data_dir))

        self.input_file = os.path.join(data_dir, 'common', 'csv', 'validate', 'malware_input_data.csv')
        self.output_file = os.path.join(data_dir, 'common', 'csv', 'validate', 'malware_output_data.csv')
        print('input_file={}, output_file={}'.format(self.input_file, self.output_file))

        self.columns_provider = CsvBrazilianProvider()

    def tearDown(self):
        duration = time.perf_counter() - self.start_time
        print(f"\n{self.id()} took {duration:.4f} seconds")

    def test_pe_preprocess_cleaner_success(self):
        headers = self.columns_provider.get_map_header_by_index()
        CsvIoValidate.validate_csv_file(self.input_file, headers)
