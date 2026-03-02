import os
import shutil
import tempfile
from pathlib import Path

from tests.utils import CsvUtil

class TempFileWriter:

    @staticmethod
    def create_temp_csv_file_from_dict_data(input_data: list[dict]) -> str:
        bytes_data = CsvUtil.csv_data_to_bytes(input_data)
        return TempFileWriter.create_temp_file_from_binary_data(bytes_data)

    @staticmethod
    def create_temp_file_from_binary_data(bytes_data: bytes) -> str:
        f = tempfile.NamedTemporaryFile(delete=False)
        try:
            f.write(bytes_data)
            f.flush()
            return f.name
        finally:
            f.close()

    @staticmethod
    def delete_file_if_exists(file_path: str) -> None:
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass

    @staticmethod
    def create_temp_dir() -> Path:
        return Path(tempfile.mkdtemp())

    @staticmethod
    def delete_temp_dir(path: Path) -> None:
        if path and path.exists():
            shutil.rmtree(path, ignore_errors=True)