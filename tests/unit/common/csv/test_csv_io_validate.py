import unittest
import time
from unittest.mock import patch
from src.common.csv import CsvIoValidate
from tests.utils import TempFileWriter
from tests.utils import CsvUtil
from tests.utils import DataProvider

class TestCsvIoValidate(unittest.TestCase):

    def setUp(self):
        self.start_time = time.perf_counter()
        self.data = DataProvider.get_csv_small()
        self.expected_headers = CsvUtil.map_index_to_headers(self.data)
        self.input_file = TempFileWriter.create_temp_csv_file_from_dict_data(self.data)
        print('input_file={}'.format(self.input_file))

    def tearDown(self):
        TempFileWriter.delete_file_if_exists(self.input_file)
        duration = time.perf_counter() - self.start_time
        print(f"\n{self.id()} took {duration:.4f} seconds")

    def test_ok_minimal_valid_csv(self):
        self.assertTrue(CsvIoValidate.validate_csv_file(self.input_file, self.expected_headers))

    def test_missing_file_raises(self):
        with patch("os.path.isfile", return_value=False):
            with self.assertRaisesRegex(ValueError, "File does not exist"):
                CsvIoValidate.validate_csv_file(self.input_file, self.expected_headers)

    def test_not_readable_raises(self):
        with patch("os.access", return_value=False):
            with self.assertRaisesRegex(ValueError, "not readable"):
                CsvIoValidate.validate_csv_file(self.input_file, self.expected_headers)

    def test_too_large_raises_without_creating_big_file(self):
        # Create a tiny file, but pretend it's huge via mocking getsize
        with patch("os.path.getsize", return_value=10_000_000):
            with self.assertRaisesRegex(ValueError, "exceeds max size"):
                CsvIoValidate.validate_csv_file(self.input_file, self.expected_headers, max_size_bytes=1)

    def test_not_utf8_raises(self):
        # Invalid UTF-8 bytes (0xFF) should trigger UnicodeDecodeError on read
        bytes_data = bytes([0xFF, 0xFE, 0x80, 0x81, 0xC0])
        test_file = TempFileWriter.create_temp_file_from_binary_data(bytes_data)
        try:
            with self.assertRaisesRegex(ValueError, "valid UTF-8"):
                CsvIoValidate.validate_csv_file(test_file, self.expected_headers)
        finally:
            TempFileWriter.delete_file_if_exists(test_file)

    def test_empty_csv_raises(self):
        bytes_data = "".encode("utf-8")
        test_file = TempFileWriter.create_temp_file_from_binary_data(bytes_data)
        try:
            with self.assertRaisesRegex(ValueError, "CSV file is empty"):
                CsvIoValidate.validate_csv_file(test_file, self.expected_headers)
        finally:
            TempFileWriter.delete_file_if_exists(test_file)

    def test_header_index_out_of_range_raises(self):
        data_0 = CsvUtil.extract_column_by_index(self.data, 0)
        test_file = TempFileWriter.create_temp_csv_file_from_dict_data(data_0)
        try:
            with self.assertRaisesRegex(ValueError, "Invalid header index"):
                CsvIoValidate.validate_csv_file(test_file, self.expected_headers)
        finally:
            TempFileWriter.delete_file_if_exists(test_file)

    def test_wrong_header_should_raise(self):
        CsvUtil.modify_column_by_index(self.data, 0, "wrong_header")
        test_file = TempFileWriter.create_temp_csv_file_from_dict_data(self.data)
        try:
            error = 'Invalid header in index 0 expected Name actual wrong_header'
            with self.assertRaisesRegex(ValueError, error):
                CsvIoValidate.validate_csv_file(test_file, self.expected_headers)
        finally:
            TempFileWriter.delete_file_if_exists(test_file)