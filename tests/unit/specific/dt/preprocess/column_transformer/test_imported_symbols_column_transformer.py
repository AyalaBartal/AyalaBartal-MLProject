import unittest
from unittest.mock import Mock
import pandas as pd

from src.specific.dt.preprocess import ImportedSymbolsColumnTransformer


class TestImportedSymbolsColumnTransformer(unittest.TestCase):
    def test_valid_transform_calls_topk_with_expected_args_and_returns_single_dataframe(self):
        # Arrange
        parse_listish = Mock()
        clean_api = Mock()
        topk = Mock()

        data = pd.DataFrame({
            "ImportedSymbols": [
                '["KERNEL32.dll!CreateFileW", "USER32.dll!MessageBoxW"]',
                '["ADVAPI32.dll!RegOpenKeyExW"]',
            ]
        })

        expected_df = pd.DataFrame({
            "api_createfilew": [1, 0],
            "api_messageboxw": [1, 0],
            "api_regopenkeyexw": [0, 1],
        })
        topk.return_value = expected_df

        transformer = ImportedSymbolsColumnTransformer(
            parse_listish=parse_listish,
            clean_api=clean_api,
            topk=topk,
            k_apis=10,
        )

        # Act
        result = transformer.valid_transform(data, "ImportedSymbols")

        # Assert
        self.assertIsInstance(result, list)
        self.assertEqual(1, len(result))
        pd.testing.assert_frame_equal(expected_df, result[0])

        topk.assert_called_once()
        args = topk.call_args[0]

        pd.testing.assert_series_equal(data["ImportedSymbols"], args[0], check_names=True)
        self.assertEqual(transformer.process_one_cell_value, args[1])
        self.assertEqual(10, args[2])
        self.assertEqual("api", args[3])

    def test_valid_transform_ignores_column_name_and_uses_imported_symbols_column(self):
        # Arrange
        topk = Mock(return_value=pd.DataFrame({"api_x": [1, 0]}))

        transformer = ImportedSymbolsColumnTransformer(
            parse_listish=Mock(),
            clean_api=Mock(),
            topk=topk,
            k_apis=3,
        )

        data = pd.DataFrame({
            "ImportedSymbols": ["a", "b"],
            "OtherColumn": ["x", "y"],
        })

        # Act
        result = transformer.valid_transform(data, "OtherColumn")

        # Assert
        self.assertEqual(1, len(result))
        pd.testing.assert_frame_equal(
            pd.DataFrame({"api_x": [1, 0]}),
            result[0],
        )
        topk.assert_called_once_with(
            data["ImportedSymbols"],
            transformer.process_one_cell_value,
            3,
            "api",
        )

    def test_valid_transform_raises_key_error_when_imported_symbols_column_missing(self):
        # Arrange
        transformer = ImportedSymbolsColumnTransformer(
            parse_listish=Mock(),
            clean_api=Mock(),
            topk=Mock(),
            k_apis=5,
        )

        data = pd.DataFrame({
            "OtherColumn": ["x", "y"]
        })

        # Act / Assert
        with self.assertRaises(KeyError):
            transformer.valid_transform(data, "OtherColumn")

    def test_process_one_cell_value_calls_parse_listish_then_clean_api(self):
        # Arrange
        parse_listish = Mock(return_value=["KERNEL32.dll!CreateFileW", "USER32.dll!MessageBoxW"])
        clean_api = Mock(return_value=["createfilew", "messageboxw"])

        transformer = ImportedSymbolsColumnTransformer(
            parse_listish=parse_listish,
            clean_api=clean_api,
            topk=Mock(),
            k_apis=5,
        )

        # Act
        result = transformer.process_one_cell_value(
            '["KERNEL32.dll!CreateFileW", "USER32.dll!MessageBoxW"]'
        )

        # Assert
        self.assertEqual(["createfilew", "messageboxw"], result)
        parse_listish.assert_called_once_with(
            '["KERNEL32.dll!CreateFileW", "USER32.dll!MessageBoxW"]'
        )
        clean_api.assert_called_once_with(
            ["KERNEL32.dll!CreateFileW", "USER32.dll!MessageBoxW"]
        )