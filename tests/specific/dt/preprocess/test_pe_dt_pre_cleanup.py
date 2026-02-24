import unittest
import os
import time

import pandas as pd

from src.common.brazilian import CsvBrazilianProvider
from src.common.precleanup import PePreprocessCleaner
from tests.utils import PathsProvider


class TestPeDtPreprocessor(unittest.TestCase):

    def setUp(self):
        self.start_time = time.perf_counter()  # Precise timing [15]

        data_dir = PathsProvider.get_test_data_dir()
        print('data_dir={}'.format(data_dir))

        self.input_file = os.path.join(data_dir, 'specific', 'dt', 'cleanup', 'malware_input_data.csv')
        self.output_file = os.path.join(data_dir, 'specific', 'dt', 'cleanup', 'malware_output_data.csv')
        print('input_file={}, output_file={}'.format(self.input_file, self.output_file))

        self.columns_provider = CsvBrazilianProvider()

    def tearDown(self):
        duration = time.perf_counter() - self.start_time
        print(f"\n{self.id()} took {duration:.4f} seconds")

    def test_pe_preprocess_cleaner_success(self):
        input_data = pd.read_csv(self.input_file)
        pe_preprocess_cleaner = PePreprocessCleaner(self.columns_provider)
        output_data = pe_preprocess_cleaner.clean(input_data)
        output_data.to_csv(self.output_file)

    def test_pre_cleanup_input_malware_csv_for_dt_success(self):
        data = pd.read_csv(self.input_file)
        sh1_list = data['SHA1'].tolist()
        self.assertEqual(10, len(sh1_list))

        # data = data.drop(columns=['Identify']) # This column missing a lot of data and it's not helpfully
        data['Identify'] = data['Identify'].fillna('unknown')

        # Remove all rows that contain any missing value (NaN, None)
        data_without_nan = data.dropna()
        sh1_list = data_without_nan['SHA1'].tolist()
        self.assertEqual(7, len(sh1_list))

        # keep only rows where all specified columns are valid integers
        integer_columns = self.columns_provider.get_non_negative_integer_headers()
        def is_even(x): return x.notna() & (x % 1 == 0)
        data = data[data[integer_columns].apply(pd.to_numeric, errors='coerce').pipe(is_even).all(axis=1)]
        sh1_list = data['SHA1'].tolist()
        self.assertEqual(9, len(sh1_list))

        # Filter rows where 'Entropy' is between 0.0 and 20.0
        float_column = self.columns_provider.get_positive_float_headers()[0]
        data = data[data[float_column].between(0.0, 20.0)]
        sh1_list = data['SHA1'].tolist()
        self.assertEqual(8, len(sh1_list))

        # keep only rows where all specified columns are valid dates
        date_columns = self.columns_provider.get_date_headers()
        data = data[data[date_columns].apply(lambda col: pd.to_datetime(col, errors='coerce').notna()).all(axis=1)]
        sh1_list = data['SHA1'].tolist()
        self.assertEqual(7, len(sh1_list))

        # Keep only rows where all specified columns have valid not empty text
        text_columns = self.columns_provider.get_text_headers()
        data = data[data[text_columns].apply(lambda s: s.notna() & s.astype(str).str.strip().ne('')).all(axis=1)]
        sh1_list = data['SHA1'].tolist()
        self.assertEqual(6, len(sh1_list))

        # Keep only rows where a  column contains valid binary values (0 or 1)
        data = data[pd.to_numeric(data['Label'], errors='coerce').isin([0, 1])]

        # Remove all rows that contain any missing value (NaN, None)
        data = data.dropna()
        sh1_list = data['SHA1'].tolist()
        self.assertEqual(2, len(sh1_list))

        data.to_csv(self.output_file)
