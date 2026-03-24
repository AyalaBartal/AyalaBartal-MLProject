import os
import unittest
from tests.utils import TempFileWriter
from src.common.image.file_io_validator import FileIoValidator


class TestFileIoValidator(unittest.TestCase):

    def setUp(self):
        self.input_dir = TempFileWriter.create_temp_dir()
        self.validator = FileIoValidator()

    def tearDown(self):
        TempFileWriter.delete_temp_dir(self.input_dir)

    # ---- validate_file_readable ----
    def test_validate_file_readable_ok(self):
        input_file = os.path.join(self.input_dir, 'a.txt')
        self.write_text_to_file(input_file)
        self.validator.validate_file_readable(str(input_file))  # should not raise

    def test_validate_file_readable_missing_raises(self):
        input_file = os.path.join(self.input_dir, 'missing.txt')
        with self.assertRaises(FileNotFoundError):
            self.validator.validate_file_readable(str(input_file))

    def test_validate_file_readable_directory_raises(self):
        with self.assertRaises(IsADirectoryError):
            self.validator.validate_file_readable(str(self.input_dir))

    # ---- validate_file_executable ----
    def test_validate_file_executable_missing_raises(self):
        input_file = os.path.join(self.input_dir, 'missing.exe')
        with self.assertRaises(FileNotFoundError):
            self.validator.validate_file_executable(str(input_file))

    def test_validate_file_executable_directory_raises(self):
        with self.assertRaises(IsADirectoryError):
            self.validator.validate_file_executable(str(self.input_dir))

    # Don’t try to create a real executable; that’s OS/CI flaky.
    # Instead, just validate the non-executable path deterministically.
    def test_validate_file_executable_not_executable_raises(self):
        input_file = os.path.join(self.input_dir, 'not.exe')
        self.write_text_to_file(input_file)
        with self.assertRaises(PermissionError):
            self.validator.validate_file_executable(str(input_file))

    # ---- validate_file_writeable ----
    def test_validate_file_writeable_ok(self):
        input_file = os.path.join(self.input_dir, 'out.txt')
        self.validator.validate_file_writeable(str(input_file))  # should not raise

    def test_validate_file_writeable_parent_missing_raises(self):
        input_file = os.path.join(self.input_dir, "no_such_dir", 'out.txt')
        with self.assertRaises(FileNotFoundError):
            self.validator.validate_file_writeable(str(input_file))

    def test_validate_file_writeable_parent_is_file_raises(self):
        parent_file = os.path.join(self.input_dir, 'out_parent.txt')
        self.write_text_to_file(parent_file)
        input_file = os.path.join(self.input_dir, 'out_parent.txt', "out.txt")
        with self.assertRaises(NotADirectoryError):
            self.validator.validate_file_writeable(str(input_file))

    def write_text_to_file(self, input_file: str):
        with open(input_file, "w", encoding="utf-8") as file:
            file.write("hello world")