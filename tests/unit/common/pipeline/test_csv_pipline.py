import unittest
import os
from typing import Callable, Any
import pandas as pd
from tempfile import TemporaryDirectory
from src.common.pipeline.csv_pipline import CsvPipline
import time


class TestDtPipeline(unittest.TestCase):

    def setUp(self):
        self.start_time = time.perf_counter()  # Precise timing [15]

    def tearDown(self):
        duration = time.perf_counter() - self.start_time
        print(f"\n{self.id()} took {duration:.4f} seconds")

    def test_pipeline_execution(self):

        with TemporaryDirectory() as tmpdir:
            start_path = os.path.join(tmpdir, "input.csv")
            end_path = os.path.join(tmpdir, "output.csv")

            # (1) Prepare dir and files
            report_dir = os.path.join(tmpdir, "report")
            os.makedirs(report_dir, exist_ok=True)
            temp_file = os.path.join(report_dir, "zero.csv")
            list_of_files = [start_path, temp_file, end_path]
            print("start={}, end={}, temp={}".format(start_path, end_path, temp_file))

            # Just to test we can write to the files.
            TestDtPipeline.write_text_to_files(list_of_files, 'word')
            TestDtPipeline.delete_files(list_of_files)

            input_data = self.get_input_data()
            input_data.to_csv(start_path, index=False)

            # (2) execute
            steps = self.build_steps()
            pipeline = CsvPipline(report_dir)
            pipeline.run(start_path, end_path, steps)

            # (3) validate
            result_df = pd.read_csv(end_path)

            output_data = self.get_output_data()

            pd.testing.assert_frame_equal(
                result_df.reset_index(drop=True),
                output_data.reset_index(drop=True)
            )

            # print("report_dir={}".format(report_dir))

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
        # 1. Lowercase column names
        conv1: Callable[[Any], Any] = lambda df: df.rename(columns=str.lower)
        # 2. Fill missing age with 0
        conv2: Callable[[Any], Any] = lambda df: df.assign(age=df["age"].fillna(0))
        # 3. Keep only valid binary Flag values (0 or 1)
        conv3: Callable[[Any], Any] = lambda df: df[pd.to_numeric(df["flag"], errors="coerce").isin([0, 1])]

        return [
            ("step1", conv1),
            ("step2", conv2),
            ("step3", conv3),
        ]

    def get_input_data(self):
        return pd.DataFrame({
            "Name": ["Alice", "Bob", "Charlie"],
            "Age": [25, None, 35],
            "Flag": ["1", "0", "invalid"]
        })

    def get_output_data(self):
        return pd.DataFrame({
            "name": ["Alice", "Bob"],
            "age": [25.0, 0.0],
            "flag": [1, 0]   # ← integers, not strings
        })
