#!/usr/bin/env python3
"""Preprocess PE CSV for Decision Tree.
- Numeric raw; Entropy clipped
- One-hot categorical
- Bitmasks expanded + raw
- Text/list Top-K counts
- Datetime parts + flags; ratios & structural flags
"""
import numpy as np
import pandas as pd

from src.specific.dt.preprocess.column_transformer_registry import ColumnTransformerRegistry


class DtPeDataTransformer:

    def __init__(self, converter):
        self.converter = converter

    # Converts raw input data into a clean numeric feature matrix ready for ML.
    # Args: top identifiers, top imported DLLs and top imported DLLs. Used bit_count is when expanding bitmask fields.
    def transform(self, data, k_ident=100, k_dlls=100, k_apis=200, bit_count=16):
        safe_num = self.converter.safe_num
        clean_dll = self.converter.clean_dll
        clean_api = self.converter.clean_api
        ratio = self.converter.ratio
        parse_listish = self.converter.parse_listish
        expand_bits = self.converter.expand_bits

        registry = ColumnTransformerRegistry(self.converter, k_dlls, k_apis, k_ident, bit_count)

        output = []
        output.append(self.calc_der(data, ratio, safe_num, clean_dll, parse_listish, clean_api))
        output.extend(self.calc_characteristics(data, bit_count, expand_bits, safe_num))

        # calc_time
        print("columns to transformer: {}".format(registry.columns()))
        for column in registry.columns():
            transformer = registry.get(column)
            output_list_of_dt = transformer.valid_transform(data, column)
            out_class_name = output_list_of_dt.__class__.__name__
            output.extend(output_list_of_dt)

        return output

    def calc_characteristics(self, data, bit_count, expand_bits, safe_num):
        output = []
        for col in ['Machine', 'PE_TYPE']:
            if col in data.columns:
                category = data[col].astype('category')
                output.append(pd.get_dummies(category, prefix=col, dummy_na=True))
        return output

    def calc_der(self, data, ratio, safe_num, clean_dll, parse_listish, clean_api):
        der = pd.DataFrame(index=data.index)
        if {'SizeOfCode', 'SizeOfImage'}.issubset(data.columns):
            der['ratio_Code_Image'] = ratio(data, 'SizeOfCode', 'SizeOfImage')
        if {'SizeOfInitializedData', 'Size'}.issubset(data.columns):
            der['ratio_InitData_Size'] = ratio(data, 'SizeOfInitializedData', 'Size')
        if {'SizeOfHeaders', 'Size'}.issubset(data.columns):
            der['ratio_Headers_Size'] = ratio(data, 'SizeOfHeaders', 'Size')
        if 'BaseOfData' in data.columns:
            der['BaseOfData_missing'] = data['BaseOfData'].isna().astype(np.int8)
        if 'PointerToSymbolTable' in data.columns:
            der['has_symtab'] = (safe_num(data['PointerToSymbolTable']) > 0).astype(np.int8)
        if 'NumberOfSymbols' in data.columns:
            der['symbols_nonzero'] = ((data['NumberOfSymbols']) > 0).astype(np.int8)
        if 'SizeOfOptionalHeader' in data.columns:
            so = safe_num(data['SizeOfOptionalHeader'])
            der['optional_header_expected'] = so.isin([224, 240]).astype(np.int8)
        if 'ImportedDlls' in data.columns:
            der['n_imported_dlls'] = data['ImportedDlls'].apply(lambda v: len(clean_dll(parse_listish(v))))
        if 'ImportedSymbols' in data.columns:
            der['n_imported_symbols'] = data['ImportedSymbols'].apply(lambda v: len(clean_api(parse_listish(v))))
        return der

