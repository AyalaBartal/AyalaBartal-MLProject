import unittest
import os
from typing import Callable, Any
from tempfile import TemporaryDirectory
from src.common.brazilian import CsvBrazilianProvider
from src.common.pipeline.csv_pipline import CsvPipline
import time

from src.common.precleanup import PePreprocessCleaner
from src.specific.dt.preprocess import \
    DtPeDataConverter, DtPeDataTransformer, DtPeDataPreprocessArgs, DtPePreprocessMapper, DtPePreprocessorProvider
from tests.utils import PathsProvider


class TestCsvPipeline(unittest.TestCase):

    def setUp(self):
        self.start_time = time.perf_counter()

        data_dir = PathsProvider.get_test_data_dir()
        print('data_dir={}'.format(data_dir))
        self.input_file = os.path.join(data_dir, 'specific', 'dt', 'pipline', 'input.csv')
        self.output_file = os.path.join(data_dir, 'specific', 'dt', 'pipline', 'output.csv')
        print('input_file={}, output_file={}'.format(self.input_file, self.output_file))

    def tearDown(self):
        duration = time.perf_counter() - self.start_time
        print(f"\n{self.id()} took {duration:.4f} seconds")

    def test_pipeline_execution(self):
        with TemporaryDirectory() as tmpdir:
            # (1) Prepare dir and files
            report_dir = os.path.join(tmpdir, "report")
            os.makedirs(report_dir, exist_ok=True)
            temp_file = os.path.join(report_dir, "zero.csv")
            print("start={}, end={}, temp={}".format(self.input_file, self.output_file, temp_file))

            # Just to test we can write to the files.
            list_of_files = [temp_file, self.output_file]
            TestCsvPipeline.write_text_to_files(list_of_files, 'word')
            TestCsvPipeline.delete_files(list_of_files)
            print("Can write and delete to files: end={} and temp={}".format(self.output_file, temp_file))

            # (2) execute
            steps = self.build_steps()
            pipeline = CsvPipline(report_dir)
            pipeline.run(self.input_file, self.output_file, steps)

    @staticmethod
    def write_text_to_files(files, text):
        for file_path in files:
            with open(file_path, 'w') as file1:
                file1.write(text)
        for out_file in files:
            assert os.path.exists(out_file)

    @staticmethod
    def delete_files(files):
        for file_path in files:
            try:
                os.remove(file_path)
                print(f"Removed: {file_path}")
            except FileNotFoundError:
                print(f"Error: {file_path} not found.")
            except PermissionError:
                print(f"Error: Permission denied to delete {file_path}.")
            except OSError as e:
                print(f"Error deleting {file_path}: {e.strerror}")

    def build_steps(self):
        # 1. Clean: filter out irrelevant columns and invalid rows
        columns_provider = CsvBrazilianProvider()
        pe_preprocess_cleaner = PePreprocessCleaner(columns_provider)
        conv1: Callable[[Any], Any] = lambda data: pe_preprocess_cleaner.clean(data)

        # 2. Map original data to format dt trainer algorithm can use
        mapper = DtPePreprocessorProvider.get_mapper()
        conv2: Callable[[Any], Any] = lambda data: mapper.map(data)

        return [
            ("cleaner", conv1),
            ("mapper", conv2),
        ]
