import unittest
from unittest.mock import Mock
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.specific.dt.preprocess import FrequencyColumnTransformer


class TestFrequencyColumnTransformer(unittest.TestCase):

    def test_valid_transform_marks_224_and_240_as_1(self):
        data = pd.DataFrame({
            'Frequency': [224, 225, 240, 0]
        })

        numeric_series = pd.Series([224, 225, 240, 0], name='Frequency')
        safe_num = Mock(return_value=numeric_series)

        transformer = FrequencyColumnTransformer(safe_num=safe_num)

        actual = transformer.valid_transform(data, 'Frequency')

        expected = pd.DataFrame({
            'Frequency': pd.Series([1, 0, 1, 0], dtype=np.int8)
        })

        self.assertEqual(1, len(actual))
        assert_frame_equal(expected, actual[0])

        safe_num.assert_called_once()
        pd.testing.assert_series_equal(data['Frequency'], safe_num.call_args[0][0])

    def test_valid_transform_marks_non_matching_values_as_0(self):
        data = pd.DataFrame({
            'Frequency': [100, 200, 300]
        })

        numeric_series = pd.Series([100, 200, 300], name='Frequency')
        transformer = FrequencyColumnTransformer(safe_num=Mock(return_value=numeric_series))

        actual = transformer.valid_transform(data, 'Frequency')

        expected = pd.DataFrame({
            'Frequency': pd.Series([0, 0, 0], dtype=np.int8)
        })

        self.assertEqual(1, len(actual))
        assert_frame_equal(expected, actual[0])

    def test_valid_transform_handles_missing_and_bad_values_after_safe_num(self):
        data = pd.DataFrame({
            'Frequency': ['224', None, 'bad', '240']
        })

        numeric_series = pd.Series([224, np.nan, np.nan, 240], name='Frequency')
        transformer = FrequencyColumnTransformer(safe_num=Mock(return_value=numeric_series))

        actual = transformer.valid_transform(data, 'Frequency')

        expected = pd.DataFrame({
            'Frequency': pd.Series([1, 0, 0, 1], dtype=np.int8)
        })

        self.assertEqual(1, len(actual))
        assert_frame_equal(expected, actual[0])