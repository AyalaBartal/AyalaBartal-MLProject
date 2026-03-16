import unittest
from unittest.mock import Mock
import pandas as pd
from pandas.testing import assert_frame_equal

from src.specific.dt.preprocess import NumberColumnTransformer


class TestNumberColumnTransformer(unittest.TestCase):
    """Tests NumberColumnTransformer.valid_transform."""

    def test_valid_transform_calls_safe_num_and_returns_dataframe(self):
        safe_num = Mock(return_value=pd.Series([1, 2, 3], name="A"))
        transformer = NumberColumnTransformer(safe_num=safe_num)

        data = pd.DataFrame({"A": ["1", "2", "3"]})

        result = transformer.valid_transform(data, "A")

        safe_num.assert_called_once_with(data["A"])
        self.assertEqual(1, len(result))

        expected = pd.DataFrame({"A": [1, 2, 3]})
        assert_frame_equal(expected, result[0])

    def test_valid_transform_preserves_column_name(self):
        safe_num = Mock(return_value=pd.Series([10, 20], name="whatever"))
        transformer = NumberColumnTransformer(safe_num=safe_num)

        data = pd.DataFrame({"MyColumn": ["10", "20"]})

        result = transformer.valid_transform(data, "MyColumn")

        self.assertEqual(["MyColumn"], list(result[0].columns))

    def test_valid_transform_handles_missing_values(self):
        safe_num = Mock(return_value=pd.Series([1.0, None, 3.0], name="X"))
        transformer = NumberColumnTransformer(safe_num=safe_num)

        data = pd.DataFrame({"X": ["1", "bad", "3"]})

        result = transformer.valid_transform(data, "X")

        expected = pd.DataFrame({"X": [1.0, None, 3.0]})
        assert_frame_equal(expected, result[0])
