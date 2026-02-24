import os
from pathlib import Path


class PathsProvider:

    @staticmethod
    def get_test_data_dir():
        test_data_dir = os.getenv("ML_MALWARE_TEST_DATA_HOME")
        if test_data_dir:
            data_path = Path(test_data_dir)
            if not data_path.exists():
                raise FileNotFoundError('TEST_DATA_HOME is set but file does not exist: {}'.format(data_path))
            if not os.path.isdir(data_path):
                raise FileNotFoundError('TEST_DATA_HOME is set but its not a dir: {}'.format(data_path))
            return str(data_path)

    @staticmethod
    def get_data_dir():
        test_dir = PathsProvider.get_target_dir("tests", 5)
        project_dir = test_dir.parent
        return os.path.join(project_dir, "data")

    @staticmethod
    def get_target_dir(target_name, max_steps):
        current_dir = Path(os.getcwd())
        for i in range(1, max_steps):
            print("index={}, dir={}", i, current_dir)
            if current_dir.name == target_name:
                return current_dir
            current_dir = current_dir.parent

