import unittest
from unittest.mock import Mock

import pandas as pd
from pandas.testing import assert_frame_equal

from src.common.precleanup import PePreprocessCleaner


class TestPePreprocessCleaner(unittest.TestCase):

    def setUp(self):
        self.columns_provider = Mock()
        self.columns_provider.get_non_negative_integer_headers.return_value = [
            "Size",
            "Machine",
        ]
        self.columns_provider.get_positive_float_headers.return_value = [
            "Entropy",
        ]
        self.columns_provider.get_date_headers.return_value = [
            "FirstSeenDate",
        ]
        self.columns_provider.get_text_headers.return_value = [
            "Identify",
            "SHA1",
        ]

        self.cleaner = PePreprocessCleaner(self.columns_provider)

    def test_clean_keeps_only_valid_rows(self):
        data = pd.DataFrame([
            {
                "Identify": "sample-a",
                "Size": 100,
                "Machine": 332,
                "Entropy": 6.5,
                "FirstSeenDate": "2024-01-01",
                "SHA1": "abc123",
                "Label": 1,
            },
            {
                "Identify": None,  # should become 'unknown' and remain valid
                "Size": 200,
                "Machine": 34404,
                "Entropy": 10.0,
                "FirstSeenDate": "2024-02-01",
                "SHA1": "def456",
                "Label": 0,
            },
            {
                "Identify": "bad-int",
                "Size": 10.5,  # invalid integer
                "Machine": 332,
                "Entropy": 5.0,
                "FirstSeenDate": "2024-03-01",
                "SHA1": "ghi789",
                "Label": 1,
            },
            {
                "Identify": "bad-float",
                "Size": 100,
                "Machine": 332,
                "Entropy": 25.0,  # out of allowed range
                "FirstSeenDate": "2024-03-01",
                "SHA1": "ghi789",
                "Label": 1,
            },
            {
                "Identify": "bad-date",
                "Size": 100,
                "Machine": 332,
                "Entropy": 7.0,
                "FirstSeenDate": "not-a-date",
                "SHA1": "ghi789",
                "Label": 1,
            },
            {
                "Identify": "   ",  # invalid text after strip
                "Size": 100,
                "Machine": 332,
                "Entropy": 7.0,
                "FirstSeenDate": "2024-03-01",
                "SHA1": "ghi789",
                "Label": 1,
            },
            {
                "Identify": "bad-label",
                "Size": 100,
                "Machine": 332,
                "Entropy": 7.0,
                "FirstSeenDate": "2024-03-01",
                "SHA1": "ghi789",
                "Label": 2,  # invalid binary label
            },
            {
                "Identify": "missing-sha1",
                "Size": 100,
                "Machine": 332,
                "Entropy": 7.0,
                "FirstSeenDate": "2024-03-01",
                "SHA1": None,
                "Label": 1,
            },
        ])

        actual = self.cleaner.clean(data.copy())

        expected = pd.DataFrame([
            {
                "Identify": "sample-a",
                "Size": 100.0,
                "Machine": 332,
                "Entropy": 6.5,
                "FirstSeenDate": "2024-01-01",
                "SHA1": "abc123",
                "Label": 1,
            },
            {
                "Identify": "unknown",
                "Size": 200.0,
                "Machine": 34404,
                "Entropy": 10.0,
                "FirstSeenDate": "2024-02-01",
                "SHA1": "def456",
                "Label": 0,
            },
        ], index=[0, 1])

        actual = actual.reset_index(drop=True)
        assert_frame_equal(actual, expected)

    def test_clean_fills_missing_identify_with_unknown(self):
        data = pd.DataFrame([
            {
                "Identify": None,
                "Size": 100,
                "Machine": 332,
                "Entropy": 5.5,
                "FirstSeenDate": "2024-01-01",
                "SHA1": "abc",
                "Label": 1,
            }
        ])

        actual = self.cleaner.clean(data.copy())

        self.assertEqual(len(actual), 1)
        self.assertEqual(actual.iloc[0]["Identify"], "unknown")

    def test_clean_removes_non_integer_rows(self):
        data = pd.DataFrame([
            {
                "Identify": "valid",
                "Size": 100,
                "Machine": 332,
                "Entropy": 5.5,
                "FirstSeenDate": "2024-01-01",
                "SHA1": "abc",
                "Label": 1,
            },
            {
                "Identify": "invalid",
                "Size": 100.1,
                "Machine": 332,
                "Entropy": 5.5,
                "FirstSeenDate": "2024-01-01",
                "SHA1": "def",
                "Label": 1,
            },
        ])

        actual = self.cleaner.clean(data.copy()).reset_index(drop=True)

        self.assertEqual(len(actual), 1)
        self.assertEqual(actual.iloc[0]["Identify"], "valid")

    def test_clean_removes_rows_with_entropy_outside_range(self):
        data = pd.DataFrame([
            {
                "Identify": "low-ok",
                "Size": 100,
                "Machine": 332,
                "Entropy": 0.0,
                "FirstSeenDate": "2024-01-01",
                "SHA1": "abc",
                "Label": 0,
            },
            {
                "Identify": "high-ok",
                "Size": 100,
                "Machine": 332,
                "Entropy": 20.0,
                "FirstSeenDate": "2024-01-02",
                "SHA1": "def",
                "Label": 1,
            },
            {
                "Identify": "too-high",
                "Size": 100,
                "Machine": 332,
                "Entropy": 20.1,
                "FirstSeenDate": "2024-01-03",
                "SHA1": "ghi",
                "Label": 1,
            },
        ])

        actual = self.cleaner.clean(data.copy()).reset_index(drop=True)

        self.assertEqual(len(actual), 2)
        self.assertEqual(list(actual["Identify"]), ["low-ok", "high-ok"])

    def test_clean_removes_rows_with_invalid_date(self):
        data = pd.DataFrame([
            {
                "Identify": "valid-date",
                "Size": 100,
                "Machine": 332,
                "Entropy": 1.0,
                "FirstSeenDate": "2024-01-01",
                "SHA1": "abc",
                "Label": 1,
            },
            {
                "Identify": "invalid-date",
                "Size": 100,
                "Machine": 332,
                "Entropy": 1.0,
                "FirstSeenDate": "bad-date",
                "SHA1": "def",
                "Label": 1,
            },
        ])

        actual = self.cleaner.clean(data.copy()).reset_index(drop=True)

        self.assertEqual(len(actual), 1)
        self.assertEqual(actual.iloc[0]["Identify"], "valid-date")

    def test_clean_removes_rows_with_empty_text(self):
        data = pd.DataFrame([
            {
                "Identify": "valid",
                "Size": 100,
                "Machine": 332,
                "Entropy": 1.0,
                "FirstSeenDate": "2024-01-01",
                "SHA1": "abc",
                "Label": 1,
            },
            {
                "Identify": "   ",
                "Size": 100,
                "Machine": 332,
                "Entropy": 1.0,
                "FirstSeenDate": "2024-01-01",
                "SHA1": "def",
                "Label": 1,
            },
        ])

        actual = self.cleaner.clean(data.copy()).reset_index(drop=True)

        self.assertEqual(len(actual), 1)
        self.assertEqual(actual.iloc[0]["Identify"], "valid")

    def test_clean_removes_rows_with_invalid_label(self):
        data = pd.DataFrame([
            {
                "Identify": "label-0",
                "Size": 100,
                "Machine": 332,
                "Entropy": 1.0,
                "FirstSeenDate": "2024-01-01",
                "SHA1": "abc",
                "Label": 0,
            },
            {
                "Identify": "label-1",
                "Size": 100,
                "Machine": 332,
                "Entropy": 1.0,
                "FirstSeenDate": "2024-01-01",
                "SHA1": "def",
                "Label": 1,
            },
            {
                "Identify": "label-2",
                "Size": 100,
                "Machine": 332,
                "Entropy": 1.0,
                "FirstSeenDate": "2024-01-01",
                "SHA1": "ghi",
                "Label": 2,
            },
            {
                "Identify": "label-text",
                "Size": 100,
                "Machine": 332,
                "Entropy": 1.0,
                "FirstSeenDate": "2024-01-01",
                "SHA1": "jkl",
                "Label": "x",
            },
        ])

        actual = self.cleaner.clean(data.copy()).reset_index(drop=True)

        self.assertEqual(len(actual), 2)
        self.assertEqual(list(actual["Identify"]), ["label-0", "label-1"])

    def test_clean_calls_columns_provider_methods(self):
        data = pd.DataFrame([
            {
                "Identify": "valid",
                "Size": 100,
                "Machine": 332,
                "Entropy": 1.0,
                "FirstSeenDate": "2024-01-01",
                "SHA1": "abc",
                "Label": 1,
            }
        ])

        self.cleaner.clean(data.copy())

        self.columns_provider.get_non_negative_integer_headers.assert_called_once_with()
        self.columns_provider.get_positive_float_headers.assert_called_once_with()
        self.columns_provider.get_date_headers.assert_called_once_with()
        self.columns_provider.get_text_headers.assert_called_once_with()
