import unittest
from unittest.mock import Mock
import pandas as pd

from src.specific.dt.preprocess import IdentifyColumnTransformer


class TestIdentifyColumnTransformer(unittest.TestCase):
    def test_valid_transform_calls_topk_with_expected_args_and_returns_dataframe_list(self):
        # Arrange
        clean_ident = Mock()
        topk = Mock()

        data = pd.DataFrame({
            "Identify": ["CreateFileW Kernel32", "OpenProcess User32"]
        })

        expected_df = pd.DataFrame({
            "id_createfilew": [1, 0],
            "id_openprocess": [0, 1],
        })
        topk.return_value = expected_df

        transformer = IdentifyColumnTransformer(
            topk=topk,
            clean_ident=clean_ident,
            k_ident=5,
        )

        # Act
        result = transformer.valid_transform(data, "Identify")

        # Assert
        self.assertIsInstance(result, list)
        self.assertEqual(1, len(result))
        pd.testing.assert_frame_equal(expected_df, result[0])

        topk.assert_called_once_with(
            data["Identify"],
            clean_ident,
            5,
            "id",
        )

    def test_valid_transform_ignores_column_name_param_and_uses_identify_column(self):
        # Arrange
        clean_ident = Mock()
        topk = Mock()

        data = pd.DataFrame({
            "Identify": ["A", "B"]
        })

        expected_df = pd.DataFrame({"id_x": [1, 0]})
        topk.return_value = expected_df

        transformer = IdentifyColumnTransformer(
            topk=topk,
            clean_ident=clean_ident,
            k_ident=3,
        )

        # Act
        result = transformer.valid_transform(data, "SomeOtherColumn")

        # Assert
        self.assertEqual(1, len(result))
        pd.testing.assert_frame_equal(expected_df, result[0])
        topk.assert_called_once_with(data["Identify"], clean_ident, 3, "id")

    def test_valid_transform_raises_key_error_when_identify_column_missing(self):
        # Arrange
        transformer = IdentifyColumnTransformer(
            topk=Mock(),
            clean_ident=Mock(),
            k_ident=10,
        )

        data = pd.DataFrame({
            "OtherColumn": ["x", "y"]
        })

        # Act / Assert
        with self.assertRaises(KeyError):
            transformer.valid_transform(data, "OtherColumn")
