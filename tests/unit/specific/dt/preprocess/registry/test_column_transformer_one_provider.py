import unittest

from src.specific.dt.preprocess.column_transformer_one_provider import ColumnTransformerOneProvider
from src.specific.dt.preprocess.pe_dt_string_converter import DtPeStringConverter
from src.specific.dt.preprocess.pe_dt_list_converter import DtPeListConverter
from src.specific.dt.preprocess.pe_dt_data_frame_converter import DtPeDataFrameConverter
from src.specific.dt.preprocess.count_apis_column_transformer import CountApisColumnTransformer
from src.specific.dt.preprocess.count_dlls_column_transformer import CountDllsColumnTransformer
from src.specific.dt.preprocess.multi_column_transformer import MultiColumnTransformer
from src.specific.dt.preprocess.frequency_column_transformer import FrequencyColumnTransformer
from src.specific.dt.preprocess.int8_column_transformer import Int8ColumnTransformer
from src.specific.dt.preprocess.missing_column_transformer import MissingColumnTransformer
from src.specific.dt.preprocess.ratio_column_transformer import RatioColumnTransformer
from src.specific.dt.preprocess.category_column_transformer import CategoryColumnTransformer
from src.specific.dt.preprocess.characteristics_column_transformer import CharacteristicsColumnTransformer
from src.specific.dt.preprocess.number_column_transformer import NumberColumnTransformer
from src.specific.dt.preprocess.entropy_column_transformer import EntropyColumnTransformer
from src.specific.dt.preprocess.identify_column_transformer import IdentifyColumnTransformer
from src.specific.dt.preprocess.imported_symbols_column_transformer import ImportedSymbolsColumnTransformer
from src.specific.dt.preprocess.first_date_column_transformer import FirstDateColumnTransformer
from src.specific.dt.preprocess.compile_time_column_transformer import CompileTimeColumnTransformer
from src.specific.dt.preprocess.imported_dlls_column_transformer import ImportedDllsColumnTransformer


class TestColumnTransformerOneProvider(unittest.TestCase):

    def setUp(self):
        self.provider = ColumnTransformerOneProvider(
            DtPeStringConverter(),
            DtPeListConverter(),
            DtPeDataFrameConverter(),
        )

    def test_get_multi(self):
        result = self.provider.get_multi({})
        self.assertIsNotNone(result)
        self.assertIsInstance(result, MultiColumnTransformer)

    def test_get_first_date(self):
        result = self.provider.get_first_date()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, FirstDateColumnTransformer)

    def test_get_compile_time(self):
        result = self.provider.get_compile_time()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, CompileTimeColumnTransformer)

    def test_get_imported_dlls(self):
        result = self.provider.get_imported_dlls(100)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, ImportedDllsColumnTransformer)

    def test_get_count_dlls(self):
        result = self.provider.get_count_dlls()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, CountDllsColumnTransformer)

    def test_get_imported_apis(self):
        result = self.provider.get_imported_apis(200)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, ImportedSymbolsColumnTransformer)

    def test_get_count_apis(self):
        result = self.provider.get_count_apis()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, CountApisColumnTransformer)

    def test_get_identify(self):
        result = self.provider.get_identify(50)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, IdentifyColumnTransformer)

    def test_get_entropy(self):
        result = self.provider.get_entropy()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, EntropyColumnTransformer)

    def test_get_characteristics(self):
        result = self.provider.get_characteristics("Chars", 16)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, CharacteristicsColumnTransformer)

    def test_get_category(self):
        result = self.provider.get_category()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, CategoryColumnTransformer)

    def test_get_number(self):
        result = self.provider.get_number()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, NumberColumnTransformer)

    def test_get_ratio(self):
        result = self.provider.get_ratio("A", "B")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, RatioColumnTransformer)

    def test_get_missing(self):
        result = self.provider.get_missing()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, MissingColumnTransformer)

    def test_get_int8(self):
        result = self.provider.get_int8()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, Int8ColumnTransformer)

    def test_get_frequency(self):
        result = self.provider.get_frequency()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, FrequencyColumnTransformer)
