from src.specific.dt.preprocess.identify_column_transformer import IdentifyColumnTransformer
from src.specific.dt.preprocess.imported_symbols_column_transformer import ImportedSymbolsColumnTransformer
from src.specific.dt.preprocess.column_transformer import ColumnTransformer
from src.specific.dt.preprocess.first_date_column_transformer import FirstDateColumnTransformer
from src.specific.dt.preprocess.compile_time_column_transformer import CompileTimeColumnTransformer
from src.specific.dt.preprocess.imported_dlls_column_transformer import ImportedDllsColumnTransformer

"""
Registry that maps column names to ColumnTransformer instances.
"""
class ColumnTransformerRegistry:

    def __init__(self, converter, k_dlls, k_apis, k_ident):
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
            'ImportedDlls': ImportedDllsColumnTransformer(parse_listish, clean_dll, topk, k_dlls),
            'ImportedSymbols': ImportedSymbolsColumnTransformer(parse_listish, clean_api, topk, k_apis),
            'Identify': IdentifyColumnTransformer(topk, clean_ident, k_ident)
        }

    # Retrieve transformer for a column. KeyError if column not registered.
    def get(self, column_name: str) -> ColumnTransformer:
        if column_name not in self.transformer_by_column:
            raise KeyError("No transformer registered for column: {}".format(column_name))
        return self.transformer_by_column[column_name]

    def contains(self, column_name: str) -> bool:
        return column_name in self.transformer_by_column

    def columns(self):
        return list(self.transformer_by_column)
