import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from src.specific.dt.preprocess import DtPeDataFrameConverter


class TestDtPeDataFrameConverter(unittest.TestCase):

    # ---------- safe_num ----------

    def test_safe_num_converts_valid_and_invalid_values(self):
        value = pd.Series(["1", "2.5", "bad", None])

        actual = DtPeDataFrameConverter.safe_num(value)

        expected = pd.Series([1.0, 2.5, 0.0, 0.0])
        assert_series_equal(actual, expected)

    # ---------- expand_bits ----------

    def test_expand_bits_basic(self):
        series = pd.Series([0, 1, 5], index=[10, 11, 12])

        actual = DtPeDataFrameConverter.expand_bits(series, n_bits=3, prefix="f")

        expected = pd.DataFrame(
            {
                "f_b0": pd.Series([0, 1, 1], index=[10, 11, 12], dtype=np.int8),
                "f_b1": pd.Series([0, 0, 0], index=[10, 11, 12], dtype=np.int8),
                "f_b2": pd.Series([0, 0, 1], index=[10, 11, 12], dtype=np.int8),
            }
        )
        assert_frame_equal(actual, expected)

    def test_expand_bits_invalid_and_missing_become_zero(self):
        series = pd.Series([None, "bad", 2])

        actual = DtPeDataFrameConverter.expand_bits(series, n_bits=2, prefix="x")

        expected = pd.DataFrame(
            {
                "x_b0": pd.Series([0, 0, 0], dtype=np.int8),
                "x_b1": pd.Series([0, 0, 1], dtype=np.int8),
            }
        )
        assert_frame_equal(actual, expected)

    # ---------- to_dt ----------

    def test_to_dt_valid_and_invalid(self):
        value = pd.Series(["2024-01-15", "bad", None])

        actual = DtPeDataFrameConverter.to_dt(value)

        self.assertEqual(str(actual.iloc[0]), "2024-01-15 00:00:00+00:00")
        self.assertTrue(pd.isna(actual.iloc[1]))
        self.assertTrue(pd.isna(actual.iloc[2]))

    # ---------- parse_tds ----------

    def test_parse_tds_unix_timestamp(self):
        series = pd.Series([0, 1710000000, 0xFFFFFFFF])

        actual_dt, actual_an = DtPeDataFrameConverter.parse_tds(series)

        expected_an = pd.Series([1, 0, 1], dtype=np.int8)
        assert_series_equal(actual_an, expected_an)

        self.assertTrue(str(actual_dt.iloc[0]).startswith("1970-01-01"))
        self.assertTrue(str(actual_dt.iloc[1]).startswith("2024-03-09"))
        self.assertTrue(str(actual_dt.iloc[2]).startswith("2106-02-07"))

    def test_parse_tds_fallback_to_string_datetime(self):
        series = pd.Series(["2024-02-03", "bad"])

        actual_dt, actual_an = DtPeDataFrameConverter.parse_tds(series)

        expected_an = pd.Series([0, 0], dtype=np.int8)
        assert_series_equal(actual_an, expected_an)

        self.assertEqual(str(actual_dt.iloc[0]), "2024-02-03 00:00:00+00:00")
        self.assertTrue(pd.isna(actual_dt.iloc[1]))

    # ---------- dt_parts ----------

    def test_dt_parts_extracts_year_month_dow(self):
        dt = pd.Series(
            pd.to_datetime(
                ["2024-01-15", None, "2024-01-21"],
                utc=True
            )
        )

        actual = DtPeDataFrameConverter.dt_parts(dt, "seen")

        expected = pd.DataFrame(
            {
                "seen_year": [2024, 0, 2024],
                "seen_month": [1, 0, 1],
                "seen_dow": [0, 0, 6],  # Monday=0, Sunday=6
            }
        )
        assert_frame_equal(actual, expected)

    # ---------- ratio ----------

    def test_ratio_basic(self):
        df = pd.DataFrame(
            {
                "a": [10, 20, 30],
                "b": [2, 5, 10],
            }
        )

        actual = DtPeDataFrameConverter.ratio(df, "a", "b")

        expected = pd.Series([5.0, 4.0, 3.0])
        assert_series_equal(actual, expected)

    def test_ratio_invalid_and_zero_division_become_zero(self):
        df = pd.DataFrame(
            {
                "a": ["10", "bad", 30, None],
                "b": [2, 5, 0, None],
            }
        )

        actual = DtPeDataFrameConverter.ratio(df, "a", "b")

        expected = pd.Series([5.0, 0.0, 0.0, 0.0])
        assert_series_equal(actual, expected)

    def test_ratio_missing_columns_returns_zero_series(self):
        df = pd.DataFrame({"x": [1, 2, 3]})

        actual = DtPeDataFrameConverter.ratio(df, "a", "b")

        expected = pd.Series([0.0, 0.0, 0.0])
        assert_series_equal(actual, expected)

    # ---------- topk ----------

    def test_topk_counts_tokens(self):
        series = pd.Series(["a b a", "b c", ""])

        def tokenizer(value):
            return str(value).split()

        actual = DtPeDataFrameConverter.topk(series, tokenizer, k=2, prefix="tok")

        expected = pd.DataFrame(
            {
                "tok_a": [2, 0, 0],
                "tok_b": [1, 1, 0],
            },
            index=series.index,
            dtype=np.int32
        )
        assert_frame_equal(actual, expected)

    def test_topk_k_zero_returns_empty_dataframe_with_same_index(self):
        series = pd.Series(["a b", "c d"], index=[5, 6])

        def tokenizer(value):
            return str(value).split()

        actual = DtPeDataFrameConverter.topk(series, tokenizer, k=0, prefix="tok")

        expected = pd.DataFrame(columns=[], index=series.index)
        assert_frame_equal(actual, expected)

    def test_topk_negative_k_returns_empty_dataframe_with_same_index(self):
        series = pd.Series(["a b", "c d"])

        def tokenizer(value):
            return str(value).split()

        actual = DtPeDataFrameConverter.topk(series, tokenizer, k=-1, prefix="tok")

        expected = pd.DataFrame(columns=[], index=series.index)
        assert_frame_equal(actual, expected)

    def test_topk_uses_tokenizer_output(self):
        series = pd.Series(["ignored", "also ignored"])

        def tokenizer(value):
            if value == "ignored":
                return ["x", "x", "y"]
            return ["y", "z"]

        actual = DtPeDataFrameConverter.topk(series, tokenizer, k=3, prefix="p")

        expected = pd.DataFrame(
            {
                "p_x": [2, 0],
                "p_y": [1, 1],
                "p_z": [0, 1],
            },
            index=series.index,
            dtype=np.int32
        )
        assert_frame_equal(actual, expected)
