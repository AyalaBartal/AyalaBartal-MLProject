import unittest
from types import SimpleNamespace
from typing import List

from src.specific.dt.preprocess import ColumnTransformerMapProvider
from src.specific.dt.preprocess.column_transformer_one_provider import ColumnTransformerOneProvider
from src.specific.dt.preprocess.pe_dt_string_converter import DtPeStringConverter
from src.specific.dt.preprocess.pe_dt_list_converter import DtPeListConverter
from src.specific.dt.preprocess.pe_dt_data_frame_converter import DtPeDataFrameConverter
from src.specific.dt.preprocess.multi_column_transformer import MultiColumnTransformer
from src.specific.dt.preprocess.category_column_transformer import CategoryColumnTransformer
from src.specific.dt.preprocess.characteristics_column_transformer import CharacteristicsColumnTransformer
from src.specific.dt.preprocess.number_column_transformer import NumberColumnTransformer
from src.specific.dt.preprocess.entropy_column_transformer import EntropyColumnTransformer
from src.specific.dt.preprocess.identify_column_transformer import IdentifyColumnTransformer
from src.specific.dt.preprocess.first_date_column_transformer import FirstDateColumnTransformer
from src.specific.dt.preprocess.compile_time_column_transformer import CompileTimeColumnTransformer


class TestColumnTransformerMapProvider(unittest.TestCase):

    def setUp(self):
        provider1 = ColumnTransformerOneProvider(DtPeStringConverter(), DtPeListConverter(), DtPeDataFrameConverter())
        self.provider = ColumnTransformerMapProvider(provider1)
        self.args = SimpleNamespace(k_dlls=10, k_apis=20, k_ident=30, bit_count=16,)

    def test_get_map_number_by_column_returns_number_transformer_for_each_numeric_column(self):
        actual = self.provider.get_map_number_by_column()

        expected_columns = TestColumnTransformerMapProvider.get_expected_columns_for_number()

        self.assertEqual(expected_columns, set(actual.keys()))

        for column in expected_columns:
            self.assertIsNotNone(actual[column])
            self.assertIsInstance(actual[column], NumberColumnTransformer)

    def test_get_map_transformer_by_column_returns_expected_transformer_type_per_column(self):
        actual = self.provider.get_map_transformer_by_column(self.args)

        expected_type_by_column = TestColumnTransformerMapProvider.get_expected_type_by_column()

        self.assertEqual(set(expected_type_by_column.keys()), set(actual.keys()))

        for column, expected_type in expected_type_by_column.items():
            self.assertIsNotNone(actual[column])
            self.assertIsInstance(actual[column], expected_type)

    def test_get_map_transformer_by_column_returns_expected_multi_transformer(self):
        actual = self.provider.get_map_transformer_by_column(self.args)

        actual_columns = {k: v for k, v in actual.items() if isinstance(v, MultiColumnTransformer)}

        expected = TestColumnTransformerMapProvider.get_expected_columns_and_transformers_for_multi()

        self.assertEqual(set(expected.keys()), set(actual_columns))

        for column, expected_names in expected.items():
            multi = actual[column]
            self.assertIsNotNone(multi)
            self.assertIsInstance(multi, MultiColumnTransformer)

            actual_names = multi.get_transformer_names()
            self.assertIsInstance(actual_names, List)
            self.assertEqual(2, len(actual_names))

            order_expected_names = sorted(expected_names)
            order_actual_names = sorted(actual_names)
            self.assertEqual(order_expected_names[0], order_actual_names[0])
            self.assertEqual(order_expected_names[1], order_actual_names[1])

    @staticmethod
    def get_expected_columns_for_number():
        return {
            'Size',
            'SizeOfImage',
            'SizeOfUninitializedData',
            'FileAlignment',
            'ImageBase',
            'BaseOfCode',
            'NumberOfSections',
            'NumberOfRvaAndSizes',
        }

    @staticmethod
    def get_expected_type_by_column():
        return {
            'FirstSeenDate': FirstDateColumnTransformer,
            'TimeDateStamp': CompileTimeColumnTransformer,
            'ImportedDlls': MultiColumnTransformer,
            'ImportedSymbols': MultiColumnTransformer,
            'Identify': IdentifyColumnTransformer,
            'Entropy': EntropyColumnTransformer,
            'Characteristics': CharacteristicsColumnTransformer,
            'DllCharacteristics': CharacteristicsColumnTransformer,
            'Machine': CategoryColumnTransformer,
            'PE_TYPE': CategoryColumnTransformer,
            'SizeOfCode': MultiColumnTransformer,
            'SizeOfInitializedData': MultiColumnTransformer,
            'SizeOfHeaders': MultiColumnTransformer,
            'BaseOfData': MultiColumnTransformer,
            'PointerToSymbolTable': MultiColumnTransformer,
            'NumberOfSymbols': MultiColumnTransformer,
            'SizeOfOptionalHeader': MultiColumnTransformer,
        }

    @staticmethod
    def get_expected_columns_and_transformers_for_multi():
        return {
            'ImportedDlls': ['ImportedDllsColumnTransformer', 'CountDllsColumnTransformer'],
            'ImportedSymbols': ['ImportedSymbolsColumnTransformer', 'CountApisColumnTransformer'],

            'SizeOfCode': ['NumberColumnTransformer', 'RatioColumnTransformer'],
            'SizeOfInitializedData': ['NumberColumnTransformer', 'RatioColumnTransformer'],
            'SizeOfHeaders': ['NumberColumnTransformer', 'RatioColumnTransformer'],

            'BaseOfData': ['NumberColumnTransformer', 'MissingColumnTransformer'],

            'PointerToSymbolTable': ['Int8ColumnTransformer', 'NumberColumnTransformer'],
            'NumberOfSymbols': ['NumberColumnTransformer', 'Int8ColumnTransformer'],
            'SizeOfOptionalHeader': ['NumberColumnTransformer', 'FrequencyColumnTransformer'],
        }
