import os
import time
import unittest
from unittest.mock import Mock, call
from src.common.image import MlIoImageWriter
from tests.utils import TempFileWriter


class TestMlIoImageWriter(unittest.TestCase):

    def setUp(self):
        self.start_time = time.perf_counter()
        self.input_dir = TempFileWriter.create_temp_dir()

        self.input_file = os.path.join(self.input_dir, 'in.dot')
        self.out_file = os.path.join(self.input_dir, 'out.jpg')
        self.dot_exe = os.path.join(self.input_dir, 'dot')

        # Dependency instances (mocks)
        self.validator = Mock()
        self.locator = Mock()
        self.executor = Mock()

        # Use Dependencies to create an instance for unit tests
        self.writer = MlIoImageWriter(self.validator, self.locator, self.executor)

    def tearDown(self):
        TempFileWriter.delete_temp_dir(self.input_dir)
        duration = time.perf_counter() - self.start_time
        print(f"\n{self.id()} took {duration:.4f} seconds")

    def test_create_jpg_from_dot_happy_path_calls_validator_and_executor(self):


        # Mock methods
        self.locator.get_dot_executable.return_value = self.dot_exe
        self.writer.create_jpg_from_dot(self.input_file, self.out_file)

        # Assert methods:
        self.locator.get_dot_executable.assert_called_once_with()
        self.validator.validate_file_readable.assert_called_once_with(self.input_file)
        self.validator.validate_file_executable.assert_called_once_with(self.dot_exe)
        self.validator.validate_file_writable.assert_called_once_with(self.out_file)

        # Executor called with exact args
        self.executor.run.assert_called_once_with(
            [str(self.dot_exe), "-Tjpg", self.input_file, "-o", self.out_file],
            check=True
        )

    def test_create_jpg_from_dot_stops_if_readable_validation_fails(self):
        self.locator.get_dot_executable.return_value = self.dot_exe
        self.validator.validate_file_readable.side_effect = ValueError("not readable")

        with self.assertRaises(ValueError):
            self.writer.create_jpg_from_dot(self.input_file, self.out_file)

        # After readable fails, nothing else should run
        self.validator.validate_file_executable.assert_not_called()
        self.validator.validate_file_writable.assert_not_called()
        self.executor.run.assert_not_called()

    def test_create_jpg_from_dot_stops_if_executable_validation_fails(self):
        self.locator.get_dot_executable.return_value = self.dot_exe
        self.validator.validate_file_executable.side_effect = ValueError("dot not executable")

        with self.assertRaises(ValueError):
            self.writer.create_jpg_from_dot(self.input_file, self.out_file)

        # Readable checked, then executable fails, then stop
        self.validator.validate_file_readable.assert_called_once_with(self.input_file)
        self.validator.validate_file_writable.assert_not_called()
        self.executor.run.assert_not_called()

    def test_create_jpg_from_dot_stops_if_writable_validation_fails(self):
        self.locator.get_dot_executable.return_value = self.dot_exe
        self.validator.validate_file_writable.side_effect = ValueError("not writable")

        with self.assertRaises(ValueError):
            self.writer.create_jpg_from_dot(self.input_file, self.out_file)

        # Readable + executable checked, then writable fails, then stop
        self.validator.validate_file_readable.assert_called_once_with(self.input_file)
        self.validator.validate_file_executable.assert_called_once_with(self.dot_exe)
        self.executor.run.assert_not_called()

    def test_create_jpg_from_dot_call_order_smoke_test(self):
        self.locator.get_dot_executable.return_value = self.dot_exe

        self.writer.create_jpg_from_dot(self.input_file, self.out_file)

        self.assertEqual(
            self.locator.mock_calls,
            [call.get_dot_executable()]
        )
        self.assertEqual(
            self.validator.mock_calls,
            [
                call.validate_file_readable(self.input_file),
                call.validate_file_executable(self.dot_exe),
                call.validate_file_writable(self.out_file),
            ]
        )
        self.assertEqual(
            self.executor.mock_calls,
            [
                call.run([str(self.dot_exe), "-Tjpg", self.input_file, "-o", self.out_file], check=True)
            ]
        )