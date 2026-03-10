import unittest
from unittest.mock import Mock
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from src.specific.dt.preprocess import CompileTimeColumnTransformer


class TestCompileTimeColumnTransformer(unittest.TestCase):
    """
    Validate that CompileTimeColumnTransformer:
    - reads data['TimeDateStamp']
    - calls parse_tds with that Series
    - calls dt_parts with parsed datetime and prefix 'TDS'
    - returns two DataFrames in the expected order
    """

    def setUp(self):
        self.parse_tds = Mock()
        self.dt_parts = Mock()

        self.transformer = CompileTimeColumnTransformer(
            parse_tds=self.parse_tds,
            dt_parts=self.dt_parts,
        )

        self.data = pd.DataFrame({
            "TimeDateStamp": [1234567890, 1234567891, None]
        })

    def test_valid_transform_returns_two_expected_dataframes(self):
        parsed_dt = pd.Series(
            pd.to_datetime(
                ["2020-01-01 10:00:00", "2020-01-02 11:00:00", None]
            ),
            name="TimeDateStamp",
        )
        anomalous = pd.Series([0, 1, 1], name="an_input_name")

        dt_parts_output = pd.DataFrame({
            "TDS_year": [2020, 2020, 0],
            "TDS_month": [1, 1, 0],
            "TDS_day": [1, 2, 0],
        })

        self.parse_tds.return_value = (parsed_dt, anomalous)
        self.dt_parts.return_value = dt_parts_output

        result = self.transformer.valid_transform(self.data, "ignored_column_name")

        self.assertEqual(2, len(result))
        self.assertIsInstance(result[0], pd.DataFrame)
        self.assertIsInstance(result[1], pd.DataFrame)

        self.parse_tds.assert_called_once()
        self.dt_parts.assert_called_once()

        parse_args = self.parse_tds.call_args[0]
        assert_series_equal(parse_args[0], self.data["TimeDateStamp"], check_names=True)

        dt_parts_args = self.dt_parts.call_args[0]
        assert_series_equal(dt_parts_args[0], parsed_dt, check_names=True)
        self.assertEqual("TDS", dt_parts_args[1])

        expected_anomalous = pd.DataFrame({
            "timestamp_anomalous": [0, 1, 1]
        })

        assert_frame_equal(result[0], dt_parts_output)
        assert_frame_equal(result[1], expected_anomalous)

    def test_valid_transform_renames_anomalous_series_to_timestamp_anomalous(self):
        parsed_dt = pd.Series(
            pd.to_datetime(["2021-05-01", None]),
            name="TimeDateStamp",
        )
        anomalous = pd.Series([0, 1], name="old_name")

        self.parse_tds.return_value = (parsed_dt, anomalous)
        self.dt_parts.return_value = pd.DataFrame({"TDS_year": [2021, 0]})

        result = self.transformer.valid_transform(
            pd.DataFrame({"TimeDateStamp": [1, 2]}),
            "any_column_name",
        )

        expected = pd.DataFrame({
            "timestamp_anomalous": [0, 1]
        })

        assert_frame_equal(result[1], expected)

    def test_valid_transform_uses_timedatestamp_column_not_column_name_argument(self):
        parsed_dt = pd.Series(pd.to_datetime(["2022-01-01"]), name="TimeDateStamp")
        anomalous = pd.Series([0], name="x")

        self.parse_tds.return_value = (parsed_dt, anomalous)
        self.dt_parts.return_value = pd.DataFrame({"TDS_year": [2022]})

        data = pd.DataFrame({
            "TimeDateStamp": [999],
            "OtherColumn": [123],
        })

        self.transformer.valid_transform(data, "OtherColumn")

        parse_args = self.parse_tds.call_args[0]
        assert_series_equal(parse_args[0], data["TimeDateStamp"], check_names=True)
