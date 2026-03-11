import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.specific.dt.preprocess import MissingColumnTransformer


class TestMissingColumnTransformer(unittest.TestCase):
    """Tests MissingColumnTransformer.valid_transform."""

    def setUp(self):
        self.transformer = MissingColumnTransformer()

    def test_valid_transform_marks_missing_values(self):
        data = pd.DataFrame({
            "A": [1, None, 3, np.nan]
        })

        result = self.transformer.valid_transform(data, "A")

        self.assertEqual(1, len(result))

        expected = pd.DataFrame({
            "A": pd.Series([0, 1, 0, 1], dtype=np.int8)
        })

        assert_frame_equal(expected, result[0])

    def test_valid_transform_all_values_present(self):
        data = pd.DataFrame({
            "B": [10, 20, 30]
        })

        result = self.transformer.valid_transform(data, "B")

        expected = pd.DataFrame({
            "B": pd.Series([0, 0, 0], dtype=np.int8)
        })

        assert_frame_equal(expected, result[0])

    def test_valid_transform_all_values_missing(self):
        data = pd.DataFrame({
            "C": [None, np.nan, None]
        })

        result = self.transformer.valid_transform(data, "C")

        expected = pd.DataFrame({
            "C": pd.Series([1, 1, 1], dtype=np.int8)
        })

        assert_frame_equal(expected, result[0])

    def test_valid_transform_preserves_column_name(self):
        data = pd.DataFrame({
            "MyColumn": [1, None]
        })

        result = self.transformer.valid_transform(data, "MyColumn")

        self.assertEqual(["MyColumn"], list(result[0].columns))
