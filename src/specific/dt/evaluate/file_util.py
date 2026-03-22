import os
from pathlib import Path


class FileUtil:

    @staticmethod
    # Checks if the given path is an existing, readable directory, and not a file.
    def validate_directory(input_dir: str, is_read, is_write):
        if not Path(input_dir).exists():
            raise FileNotFoundError('Not found file: {}'.format(input_dir))
        if not os.path.isdir(input_dir):
            raise FileNotFoundError('Not dir: {}'.format(input_dir))
        if is_read and not os.access(input_dir, os.R_OK):
            raise FileNotFoundError('Not readable file: {}'.format(input_dir))
        if is_write and not os.access(input_dir, os.W_OK):
            raise FileNotFoundError('Not writeable file: {}'.format(input_dir))
