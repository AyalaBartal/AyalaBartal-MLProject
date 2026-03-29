from src.common.preprocessor import DtPeDataFrameConverter, DtPeListConverter, DtPeStringConverter
from src.common.preprocessor.count_apis_column_transformer import CountApisColumnTransformer
from src.common.preprocessor.count_dlls_column_transformer import CountDllsColumnTransformer
from src.common.preprocessor.multi_column_transformer import MultiColumnTransformer
from src.common.preprocessor.frequency_column_transformer import FrequencyColumnTransformer
from src.common.preprocessor.int8_column_transformer import Int8ColumnTransformer
from src.common.preprocessor.missing_column_transformer import MissingColumnTransformer
from src.common.preprocessor.ratio_column_transformer import RatioColumnTransformer
from src.common.preprocessor.category_column_transformer import CategoryColumnTransformer
from src.common.preprocessor.characteristics_column_transformer import CharacteristicsColumnTransformer
from src.common.preprocessor.number_column_transformer import NumberColumnTransformer
from src.common.preprocessor.entropy_column_transformer import EntropyColumnTransformer
from src.common.preprocessor.identify_column_transformer import IdentifyColumnTransformer
from src.common.preprocessor.imported_symbols_column_transformer import ImportedSymbolsColumnTransformer
from src.common.preprocessor.column_transformer import ColumnTransformer
from src.common.preprocessor.first_date_column_transformer import FirstDateColumnTransformer
from src.common.preprocessor.compile_time_column_transformer import CompileTimeColumnTransformer
from src.common.preprocessor.imported_dlls_column_transformer import ImportedDllsColumnTransformer


class ColumnTransformerOneProvider:

    def __init__(self,
                 str_converter: DtPeStringConverter,
                 list_converter: DtPeListConverter,
                 df_converter: DtPeDataFrameConverter):

        # str_converter
        self.parse_listish = str_converter.parse_listish
        self.clean_ident = str_converter.clean_ident

        # list_converter
        self.clean_dll = list_converter.clean_dlls
        self.clean_api = list_converter.clean_apis

        # df_converter
        self.safe_num = df_converter.safe_num
        self.dt_parts = df_converter.dt_parts
        self.ratio = df_converter.ratio
        self.expand_bits = df_converter.expand_bits
        self.topk = df_converter.topk
        self.parse_tds = df_converter.parse_tds
        self.to_dt = df_converter.to_dt

    def get_multi(self, transformer_by_name) -> ColumnTransformer:
        return MultiColumnTransformer(transformer_by_name)

    def get_first_date(self) -> ColumnTransformer:
        return FirstDateColumnTransformer(self.dt_parts, self.to_dt)

    def get_compile_time(self) -> ColumnTransformer:
        return CompileTimeColumnTransformer(self.parse_tds, self.dt_parts)

    def get_imported_dlls(self, top_k_dlls) -> ColumnTransformer:
        return ImportedDllsColumnTransformer(self.parse_listish, self.clean_dll, self.topk, top_k_dlls)

    def get_count_dlls(self) -> ColumnTransformer:
        return CountDllsColumnTransformer(self.parse_listish, self.clean_dll)

    def get_imported_apis(self, top_k_apis) -> ColumnTransformer:
        return ImportedSymbolsColumnTransformer(self.parse_listish, self.clean_api, self.topk, top_k_apis)

    def get_count_apis(self) -> ColumnTransformer:
        return CountApisColumnTransformer(self.parse_listish, self.clean_api)

    def get_identify(self, k_ident) -> ColumnTransformer:
        return IdentifyColumnTransformer(self.topk, self.clean_ident, k_ident)

    def get_entropy(self) -> ColumnTransformer:
        return EntropyColumnTransformer(self.safe_num)

    def get_characteristics(self, prefix, bit_count) -> ColumnTransformer:
        return CharacteristicsColumnTransformer(self.expand_bits, self.safe_num, prefix, bit_count)

    def get_category(self) -> ColumnTransformer:
        return CategoryColumnTransformer()

    def get_number(self) -> ColumnTransformer:
        return NumberColumnTransformer(self.safe_num)

    def get_ratio(self, a, b) -> ColumnTransformer:
        return RatioColumnTransformer(self.ratio, a, b)

    def get_missing(self) -> ColumnTransformer:
        return MissingColumnTransformer()

    def get_int8(self) -> ColumnTransformer:
        return Int8ColumnTransformer(self.safe_num)

    def get_frequency(self) -> ColumnTransformer:
        return FrequencyColumnTransformer(self.safe_num)
