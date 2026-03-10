import unittest
from unittest.mock import Mock
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from src.specific.dt.preprocess import CharacteristicsColumnTransformer


class TestCharacteristicsColumnTransformer(unittest.TestCase):
    """
    Validate that CharacteristicsColumnTransformer:
    - returns two DataFrames
    - calls expand_bits with the raw source column
    - calls safe_num with the raw source column
    - builds the expected '<prefix>_raw' DataFrame
    """

    def setUp(self):
        self.expand_bits = Mock()
        self.safe_num = Mock()

        self.prefix = "Characteristics"
        self.bit_count = 4

        self.transformer = CharacteristicsColumnTransformer(
            expand_bits=self.expand_bits,
            safe_num=self.safe_num,
            prefix=self.prefix,
            bit_count=self.bit_count,
        )

        self.data = pd.DataFrame({
            "DllCharacteristics": [5, 2, None]
        })
        self.column_name = "DllCharacteristics"

    def test_valid_transform_returns_two_expected_dataframes(self):
        part_1 = pd.DataFrame({
            "Characteristics_bit_0": [1, 0, 0],
            "Characteristics_bit_1": [0, 1, 0],
            "Characteristics_bit_2": [1, 0, 0],
            "Characteristics_bit_3": [0, 0, 0],
        })

        safe_num_output = pd.Series([5.0, 2.0, 0.0], name=self.column_name)

        self.expand_bits.return_value = part_1
        self.safe_num.return_value = safe_num_output

        result = self.transformer.valid_transform(self.data, self.column_name)

        self.assertEqual(2, len(result))
        self.assertIsInstance(result[0], pd.DataFrame)
        self.assertIsInstance(result[1], pd.DataFrame)

        self.expand_bits.assert_called_once()
        self.safe_num.assert_called_once()

        expand_args = self.expand_bits.call_args[0]
        assert_series_equal(expand_args[0], self.data[self.column_name], check_names=True)
        self.assertEqual(self.bit_count, expand_args[1])
        self.assertEqual(self.prefix, expand_args[2])

        safe_num_args = self.safe_num.call_args[0]
        assert_series_equal(safe_num_args[0], self.data[self.column_name], check_names=True)

        expected_raw = pd.DataFrame({
            f"{self.prefix}_raw": [5.0, 2.0, 0.0]
        })

        assert_frame_equal(result[0], part_1)
        assert_frame_equal(result[1], expected_raw)

    def test_get_part_1_returns_expand_bits_result(self):
        expected = pd.DataFrame({
            "Characteristics_bit_0": [1, 0, 1]
        })
        self.expand_bits.return_value = expected

        result = self.transformer.get_part_1(self.data, self.column_name)

        self.expand_bits.assert_called_once()
        expand_args = self.expand_bits.call_args[0]
        assert_series_equal(expand_args[0], self.data[self.column_name], check_names=True)
        self.assertEqual(self.bit_count, expand_args[1])
        self.assertEqual(self.prefix, expand_args[2])

        assert_frame_equal(result, expected)

    def test_get_part_2_returns_renamed_dataframe_from_safe_num(self):
        safe_num_output = pd.Series([5.0, 2.0, 0.0], name="anything")
        self.safe_num.return_value = safe_num_output

        result = self.transformer.get_part_2(self.data, self.column_name)

        self.safe_num.assert_called_once()
        safe_num_args = self.safe_num.call_args[0]
        assert_series_equal(safe_num_args[0], self.data[self.column_name], check_names=True)

        expected = pd.DataFrame({
            f"{self.prefix}_raw": [5.0, 2.0, 0.0]
        })

        assert_frame_equal(result, expected)
