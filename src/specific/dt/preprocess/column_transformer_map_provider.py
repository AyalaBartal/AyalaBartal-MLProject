from typing import Dict

from src.specific.dt.preprocess.column_transformer_one_provider import ColumnTransformerOneProvider
from src.specific.dt.preprocess.column_transformer import ColumnTransformer


class ColumnTransformerMapProvider:

    NUMERIC_COLUMNS = ['Size', 'SizeOfImage', 'SizeOfUninitializedData', 'FileAlignment', 'ImageBase', 'BaseOfCode',
                       'NumberOfSections', 'NumberOfRvaAndSizes']

    def __init__(self, transformer_one_provider: ColumnTransformerOneProvider):
        self.provider = transformer_one_provider

    def get_map_number_by_column(self) -> Dict[str, ColumnTransformer]:
        transformer_by_column = {}
        for column in self.NUMERIC_COLUMNS:
            transformer_by_column[column] = self.provider.get_number()
        return transformer_by_column

    def get_map_transformer_by_column(self, args) -> Dict[str, ColumnTransformer]:
        return {
            'FirstSeenDate': self.provider.get_first_date(),
            'TimeDateStamp': self.provider.get_compile_time(),
            'ImportedDlls': self.provider.get_multi({
                '{}':  self.provider.get_imported_dlls(args.k_dlls),
                'n_imported_dlls': self.provider.get_count_dlls()
             }),
            'ImportedSymbols': self.provider.get_multi({
                '{}': self.provider.get_imported_apis(args.k_apis),
                'n_imported_symbols': self.provider.get_count_apis()
            }),
            'Identify': self.provider.get_identify(args.k_ident),
            'Entropy': self.provider.get_entropy(),
            'Characteristics': self.provider.get_characteristics('char', args.bit_count),
            'DllCharacteristics': self.provider.get_characteristics('dllc', args.bit_count),
            'Machine': self.provider.get_category(),
            'PE_TYPE': self.provider.get_category(),
            'SizeOfCode': self.provider.get_multi({
                '{}': self.provider.get_number(),
                'ratio_Code_Image': self.provider.get_ratio('SizeOfCode', 'SizeOfImage')
            }),
            'SizeOfInitializedData': self.provider.get_multi({
                '{}': self.provider.get_number(),
                'ratio_InitData_Size': self.provider.get_ratio('SizeOfInitializedData', 'Size')
            }),
            'SizeOfHeaders': self.provider.get_multi({
                '{}': self.provider.get_number(),
                'ratio_Headers_Size': self.provider.get_ratio('SizeOfHeaders', 'Size')
            }),
            'BaseOfData': self.provider.get_multi({
                '{}': self.provider.get_number(),
                '{}_missing': self.provider.get_missing()
            }),
            'PointerToSymbolTable': self.provider.get_multi({
                'has_symtab': self.provider.get_int8(),
                '{}': self.provider.get_number(),
            }),
            'NumberOfSymbols': self.provider.get_multi({
                '{}': self.provider.get_number(),
                'symbols_nonzero':self.provider.get_int8(),
            }),
            'SizeOfOptionalHeader': self.provider.get_multi({
                '{}': self.provider.get_number(),
                'optional_header_expected': self.provider.get_frequency()
            }),
        }
