import unittest
from unittest.mock import Mock
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.specific.dt.preprocess import FirstDateColumnTransformer


class TestFirstDateColumnTransformer(unittest.TestCase):

    def test_valid_transform_returns_dt_parts_and_missing_flag(self):
        data = pd.DataFrame({
            'FirstSeenDate': ['2024-01-10', None, 'bad-date']
        })

        dt_series = pd.Series(
            [
                pd.Timestamp('2024-01-10'),
                pd.NaT,
                pd.NaT,
            ],
            name='FirstSeenDate'
        )

        dt_parts_output = pd.DataFrame({
            'FirstSeen_year': [2024, 0, 0],
            'FirstSeen_month': [1, 0, 0],
            'FirstSeen_dow': [2, 0, 0],
        })

        to_dt = Mock(return_value=dt_series)
        dt_parts = Mock(return_value=dt_parts_output)

        transformer = FirstDateColumnTransformer(dt_parts=dt_parts, to_dt=to_dt)

        actual = transformer.valid_transform(data, column_name='ignored')

        self.assertEqual(2, len(actual))

        expected_missing = pd.DataFrame({
            'FirstSeen_missing': pd.Series([0, 1, 1], dtype=np.int8)
        })

        assert_frame_equal(dt_parts_output, actual[0])
        assert_frame_equal(expected_missing, actual[1])

        to_dt.assert_called_once()
        dt_parts.assert_called_once()

        to_dt_arg = to_dt.call_args[0][0]
        pd.testing.assert_series_equal(data['FirstSeenDate'], to_dt_arg)

        dt_parts.assert_called_once_with(dt_series, 'FirstSeen')

    def test_valid_transform_when_all_dates_exist(self):
        data = pd.DataFrame({
            'FirstSeenDate': ['2024-01-10', '2024-01-11']
        })

        dt_series = pd.Series(
            [
                pd.Timestamp('2024-01-10'),
                pd.Timestamp('2024-01-11'),
            ],
            name='FirstSeenDate'
        )

        dt_parts_output = pd.DataFrame({
            'FirstSeen_year': [2024, 2024]
        })

        transformer = FirstDateColumnTransformer(
            dt_parts=Mock(return_value=dt_parts_output),
            to_dt=Mock(return_value=dt_series)
        )

        actual = transformer.valid_transform(data, column_name='ignored')

        expected_missing = pd.DataFrame({
            'FirstSeen_missing': pd.Series([0, 0], dtype=np.int8)
        })

        self.assertEqual(2, len(actual))
        assert_frame_equal(dt_parts_output, actual[0])
        assert_frame_equal(expected_missing, actual[1])

    def test_valid_transform_when_all_dates_missing(self):
        data = pd.DataFrame({
            'FirstSeenDate': [None, None]
        })

        dt_series = pd.Series([pd.NaT, pd.NaT], name='FirstSeenDate')

        dt_parts_output = pd.DataFrame({
            'FirstSeen_year': [0, 0]
        })

        transformer = FirstDateColumnTransformer(
            dt_parts=Mock(return_value=dt_parts_output),
            to_dt=Mock(return_value=dt_series)
        )

        actual = transformer.valid_transform(data, column_name='ignored')

        expected_missing = pd.DataFrame({
            'FirstSeen_missing': pd.Series([1, 1], dtype=np.int8)
        })

        self.assertEqual(2, len(actual))
        assert_frame_equal(dt_parts_output, actual[0])
        assert_frame_equal(expected_missing, actual[1])
