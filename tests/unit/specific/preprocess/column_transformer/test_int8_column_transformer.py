import unittest
from unittest.mock import Mock
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.specific.dt.preprocess import Int8ColumnTransformer


class TestInt8ColumnTransformer(unittest.TestCase):
    """Tests Int8ColumnTransformer.valid_transform."""

    def test_valid_transform_converts_positive_numbers_to_1_and_others_to_0(self):
        safe_num = Mock(return_value=pd.Series([5, 0, -3, 2], name="MyColumn"))
        transformer = Int8ColumnTransformer(safe_num=safe_num)

        data = pd.DataFrame({
            "MyColumn": ["a", "b", "c", "d"]
        })

        result = transformer.valid_transform(data, "MyColumn")

        safe_num.assert_called_once_with(data["MyColumn"])
        self.assertEqual(1, len(result))

        expected = pd.DataFrame({
            "MyColumn": pd.Series([1, 0, 0, 1], dtype=np.int8)
        })

        assert_frame_equal(expected, result[0])

    def test_valid_transform_returns_int8_dataframe(self):
        safe_num = Mock(return_value=pd.Series([1, -1, 0], name="Flag"))
        transformer = Int8ColumnTransformer(safe_num=safe_num)

        data = pd.DataFrame({
            "Flag": [10, 20, 30]
        })

        result = transformer.valid_transform(data, "Flag")

        actual_df = result[0]
        self.assertEqual(np.int8, actual_df["Flag"].dtype)
        self.assertEqual([1, 0, 0], actual_df["Flag"].tolist())

    def test_valid_transform_uses_requested_column_name_for_output(self):
        safe_num = Mock(return_value=pd.Series([7, -2], name="Anything"))
        transformer = Int8ColumnTransformer(safe_num=safe_num)

        data = pd.DataFrame({
            "SomeColumn": ["x", "y"]
        })

        result = transformer.valid_transform(data, "SomeColumn")

        self.assertEqual(["SomeColumn"], list(result[0].columns))

    def test_valid_transform_with_all_non_positive_values_returns_all_zeros(self):
        safe_num = Mock(return_value=pd.Series([0, -1, -100], name="Score"))
        transformer = Int8ColumnTransformer(safe_num=safe_num)

        data = pd.DataFrame({
            "Score": [1, 2, 3]
        })

        result = transformer.valid_transform(data, "Score")

        expected = pd.DataFrame({
            "Score": pd.Series([0, 0, 0], dtype=np.int8)
        })

        assert_frame_equal(expected, result[0])
