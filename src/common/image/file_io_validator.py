import os
import tempfile
from pathlib import Path


class FileIoValidator:

    # Validate that file exists and is readable.
    def validate_file_readable(self, file_path: str) -> None:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError("File does not exist: {}".format(file_path))

        if not path.is_file():
            raise IsADirectoryError("Not a file: {}".format(file_path))
        try:
            with path.open("r", encoding="utf-8"):
                pass
        except OSError as e:
            raise PermissionError("File is not readable {}".format(file_path))

    # Validate that file exists and can be executed.
    def validate_file_executable(self, file_path: str) -> None:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError("File does not exist: {}".format(file_path))

        if not path.is_file():
            raise IsADirectoryError("Not a file: {}".format(file_path))

        if not os.access(path, os.X_OK):
            raise PermissionError("File is not executable: {}".format(file_path))

    # Validate that file can be written to even if it not exists yet.
    def validate_file_writeable(self,  file_path: str) -> None:
        path = Path(file_path)
        directory = path.parent

        if not directory.exists():
            raise FileNotFoundError(f"Directory does not exist: {directory}")

        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        try:
            with tempfile.TemporaryFile(dir=directory):
                pass
        except OSError as e:
            raise PermissionError(f"Directory not writable: {directory}") from e