import unittest
from unittest.mock import Mock
import pandas as pd
from pandas.testing import assert_frame_equal

from src.specific.dt.preprocess import RatioColumnTransformer


class TestRatioColumnTransformer(unittest.TestCase):

    def test_valid_transform_returns_ratio_dataframe_when_both_columns_exist(self):
        ratio = Mock(return_value=pd.Series([0.5, 2.0], index=[0, 1]))
        transformer = RatioColumnTransformer(
            ratio=ratio,
            column_a="Size",
            column_b="BaseOfCode",
        )

        data = pd.DataFrame({
            "Size": [100, 200],
            "BaseOfCode": [200, 100],
            "Other": [1, 2],
        })

        result = transformer.valid_transform(data, "ignored")

        self.assertEqual(1, len(result))
        ratio.assert_called_once_with(data, "Size", "BaseOfCode")

        expected = pd.DataFrame({
            "Size_BaseOfCode_ratio": [0.5, 2.0]
        })
        assert_frame_equal(expected, result[0])

    def test_valid_transform_returns_empty_list_when_first_column_missing(self):
        ratio = Mock()
        transformer = RatioColumnTransformer(
            ratio=ratio,
            column_a="Size",
            column_b="BaseOfCode",
        )

        data = pd.DataFrame({
            "BaseOfCode": [200, 100],
        })

        result = transformer.valid_transform(data, "ignored")

        self.assertEqual([], result)
        ratio.assert_not_called()

    def test_valid_transform_returns_empty_list_when_second_column_missing(self):
        ratio = Mock()
        transformer = RatioColumnTransformer(
            ratio=ratio,
            column_a="Size",
            column_b="BaseOfCode",
        )

        data = pd.DataFrame({
            "Size": [100, 200],
        })

        result = transformer.valid_transform(data, "ignored")

        self.assertEqual([], result)
        ratio.assert_not_called()

    def test_valid_transform_uses_expected_output_column_name(self):
        ratio = Mock(return_value=pd.Series([1.25]))
        transformer = RatioColumnTransformer(
            ratio=ratio,
            column_a="A",
            column_b="B",
        )

        data = pd.DataFrame({
            "A": [5],
            "B": [4],
        })

        result = transformer.valid_transform(data, "ignored")

        self.assertEqual(["A_B_ratio"], list(result[0].columns))

    def test_valid_transform_preserves_index(self):
        ratio = Mock(return_value=pd.Series([0.1, 0.2], index=["x", "y"]))
        transformer = RatioColumnTransformer(
            ratio=ratio,
            column_a="A",
            column_b="B",
        )

        data = pd.DataFrame({
            "A": [1, 2],
            "B": [10, 10],
        }, index=["x", "y"])

        result = transformer.valid_transform(data, "ignored")

        expected = pd.DataFrame({
            "A_B_ratio": [0.1, 0.2]
        }, index=["x", "y"])
        assert_frame_equal(expected, result[0])
