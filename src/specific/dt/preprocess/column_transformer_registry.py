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
from src.specific.dt.preprocess.column_transformer import ColumnTransformer
from src.specific.dt.preprocess.first_date_column_transformer import FirstDateColumnTransformer
from src.specific.dt.preprocess.compile_time_column_transformer import CompileTimeColumnTransformer
from src.specific.dt.preprocess.imported_dlls_column_transformer import ImportedDllsColumnTransformer


# Registry that maps column names to ColumnTransformer instances.
class ColumnTransformerRegistry:

    NUMERIC_COLUMNS = ['Size', 'SizeOfImage', 'SizeOfUninitializedData', 'FileAlignment', 'ImageBase', 'BaseOfCode',
                       'NumberOfSections', 'NumberOfRvaAndSizes']

    # Args: top identifiers, top imported DLLs and top imported DLLs. Used bit_count is when expanding bitmask fields.
    def __init__(self, converter, k_dlls, k_apis, k_ident, bit_count):
        self.converter = converter

        safe_num = self.converter.safe_num
        clean_dll = self.converter.clean_dll
        clean_api = self.converter.clean_api
        dt_parts = self.converter.dt_parts
        ratio = self.converter.ratio
        parse_listish = self.converter.parse_listish
        expand_bits = self.converter.expand_bits
        topk = self.converter.topk
        clean_ident = self.converter.clean_ident
        parse_tds = self.converter.parse_tds
        to_dt = self.converter.to_dt

        self.transformer_by_column = {
            'FirstSeenDate': FirstDateColumnTransformer(dt_parts, to_dt),
            'TimeDateStamp': CompileTimeColumnTransformer(parse_tds, dt_parts),
            'ImportedDlls': MultiColumnTransformer({
                '{}': ImportedDllsColumnTransformer(parse_listish, clean_dll, topk, k_dlls),
                'n_imported_dlls': CountDllsColumnTransformer(parse_listish, clean_dll)
             }),
            'ImportedSymbols': MultiColumnTransformer({
                '{}': ImportedSymbolsColumnTransformer(parse_listish, clean_api, topk, k_apis),
                'n_imported_symbols': CountApisColumnTransformer(parse_listish, clean_api)
            }),
            'Identify': IdentifyColumnTransformer(topk, clean_ident, k_ident),
            'Entropy': EntropyColumnTransformer(safe_num),
            'Characteristics': CharacteristicsColumnTransformer(expand_bits, safe_num, 'char', bit_count),
            'DllCharacteristics': CharacteristicsColumnTransformer(expand_bits, safe_num, 'dllc', bit_count),
            'Machine': CategoryColumnTransformer(),
            'PE_TYPE': CategoryColumnTransformer(),
            'SizeOfCode': MultiColumnTransformer({
                '{}': NumberColumnTransformer(safe_num),
                'ratio_Code_Image': RatioColumnTransformer(ratio, 'SizeOfCode', 'SizeOfImage')
            }),
            'SizeOfInitializedData': MultiColumnTransformer({
                '{}': NumberColumnTransformer(safe_num),
                'ratio_InitData_Size': RatioColumnTransformer(ratio, 'SizeOfInitializedData', 'Size')
            }),
            'SizeOfHeaders': MultiColumnTransformer({
                '{}': NumberColumnTransformer(safe_num),
                'ratio_Headers_Size': RatioColumnTransformer(ratio, 'SizeOfHeaders', 'Size')
            }),
            'BaseOfData': MultiColumnTransformer({
                '{}': NumberColumnTransformer(safe_num),
                '{}_missing': MissingColumnTransformer()
            }),
            'PointerToSymbolTable': MultiColumnTransformer({
                'has_symtab': Int8ColumnTransformer(safe_num),
                '{}': NumberColumnTransformer(safe_num)
            }),
            'NumberOfSymbols': MultiColumnTransformer({
                '{}': NumberColumnTransformer(safe_num),
                'symbols_nonzero': Int8ColumnTransformer(safe_num)
            }),
            'SizeOfOptionalHeader': MultiColumnTransformer({
                '{}': NumberColumnTransformer(safe_num),
                'optional_header_expected': FrequencyColumnTransformer(safe_num)
            }),
        }

        for column in self.NUMERIC_COLUMNS:
            self.transformer_by_column[column] = NumberColumnTransformer(safe_num)

    # Retrieve transformer for a column. KeyError if column not registered.
    def get(self, column_name: str) -> ColumnTransformer:
        if column_name not in self.transformer_by_column:
            raise KeyError("No transformer registered for column: {}".format(column_name))
        return self.transformer_by_column[column_name]

    def contains(self, column_name: str) -> bool:
        return column_name in self.transformer_by_column

    def columns(self):
        return list(self.transformer_by_column)
