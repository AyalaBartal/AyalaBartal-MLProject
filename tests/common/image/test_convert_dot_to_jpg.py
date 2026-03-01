import pytest
import unittest
import os
import time

from src.common.image.ml_io_image import MlIoImageWriter
from tests.utils import PathsProvider


class TestPeDtPreprocessor(unittest.TestCase):

    def setUp(self):
        self.start_time = time.perf_counter()  # Precise timing [15]

    def tearDown(self):
        duration = time.perf_counter() - self.start_time
        print(f"\n{self.id()} took {duration:.4f} seconds")

    @pytest.mark.skip(reason="Requires Graphviz dot executable - optional for deployment")
    @pytest.mark.skip(reason="Requires Graphviz")
    def test_dot_to_jpg(self):
        data_dir = PathsProvider.get_test_data_dir()
        input_file = os.path.join(data_dir, 'specific', 'dt', 'train', 'output', 'dt_report', 'decision_tree_model.dot')
        out_file = os.path.join(data_dir, 'specific', 'dt', 'train', 'output', 'dt_report', 'decision_tree_model.jpg')
        MlIoImageWriter.create_jpg_from_dot(input_file, out_file)
