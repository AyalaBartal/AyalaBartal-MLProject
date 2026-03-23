import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.common.validator.file_validator import FileValidator


class TestFileUtil(unittest.TestCase):

    def test_is_readable_directory_valid_directory_no_exception(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("os.access", return_value=True):
                FileValidator.validate_directory(temp_dir, is_read=True, is_write=True)

    def test_is_readable_directory_raises_when_path_not_exists(self):
        missing_dir = str(Path(tempfile.gettempdir()) / "definitely_missing_test_dir_12345")

        with self.assertRaises(FileNotFoundError) as context:
            FileValidator.validate_directory(missing_dir, is_read=False, is_write=False)

        self.assertEqual(f"Not found file: {missing_dir}", str(context.exception))

    def test_is_readable_directory_raises_when_path_is_file(self):
        with tempfile.NamedTemporaryFile() as temp_file:
            with self.assertRaises(FileNotFoundError) as context:
                FileValidator.validate_directory(temp_file.name, is_read=False, is_write=False)

        self.assertEqual(f"Not dir: {temp_file.name}", str(context.exception))

    def test_is_readable_directory_raises_when_not_readable_and_read_required(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("os.access", side_effect=lambda path, mode: mode != os.R_OK):
                with self.assertRaises(FileNotFoundError) as context:
                    FileValidator.validate_directory(temp_dir, is_read=True, is_write=False)

        self.assertEqual(f"Not readable file: {temp_dir}", str(context.exception))

    def test_is_readable_directory_raises_when_not_writable_and_write_required(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("os.access", side_effect=lambda path, mode: mode != os.W_OK):
                with self.assertRaises(FileNotFoundError) as context:
                    FileValidator.validate_directory(temp_dir, is_read=False, is_write=True)

        self.assertEqual(f"Not writeable file: {temp_dir}", str(context.exception))

    def test_is_readable_directory_does_not_check_read_when_is_read_false(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("os.access", return_value=False):
                FileValidator.validate_directory(temp_dir, is_read=False, is_write=False)

    def test_is_readable_directory_does_not_check_write_when_is_write_false(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("os.access", return_value=True):
                FileValidator.validate_directory(temp_dir, is_read=True, is_write=False)
