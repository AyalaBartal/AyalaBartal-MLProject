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
        topk = self.converter.topk
        clean_ident = self.converter.clean_ident

        registry = ColumnTransformerRegistry(self.converter)

        output = []
        output.extend(self.calc_nums(data, safe_num))
        output.append(self.calc_der(data, ratio, safe_num, clean_dll, parse_listish, clean_api))
        output.extend(self.calc_characteristics(data, bit_count, expand_bits, safe_num))
        output.extend(self.calc_identify(clean_ident, data, k_ident, topk))
        output.extend(self.calc_dlls(data, k_apis, k_dlls, topk, clean_dll, clean_api, parse_listish))

        # calc_time
        for column in registry.columns():
            transformer = registry.get(column)
            output.extend(transformer.valid_transform(data, column))

        return output

    def calc_dlls(self, data, k_apis, k_dlls, topk, clean_dll, clean_api, parse_listish):
        output = []
        if 'ImportedDlls' in data.columns:
            output.append(topk(data['ImportedDlls'], lambda v: clean_dll(parse_listish(v)), k_dlls, 'dll'))
        if 'ImportedSymbols' in data.columns:
            output.append(topk(data['ImportedSymbols'], lambda v: clean_api(parse_listish(v)), k_apis, 'api'))
        return output

    def calc_identify(self, clean_ident, data, k_ident, topk):
        output2 = []
        if 'Identify' in data.columns:
            output2.append(topk(data['Identify'], clean_ident, k_ident, 'id'))
        return output2

    def calc_characteristics(self, data, bit_count, expand_bits, safe_num):
        output = []
        for col, p in [('Characteristics', 'char'), ('DllCharacteristics', 'dllc')]:
            if col in data.columns:
                if expand_bits:
                    value0 = data[col]
                    part1 = expand_bits(value0, bit_count, p)
                    output.append(part1)
                value1 = data[col]
                value2 = safe_num(value1)
                value3 = value2.rename(f"{p}_raw").to_frame()
                output.append(value3)
        for col in ['Machine', 'PE_TYPE']:
            if col in data.columns:
                category = data[col].astype('category')
                output.append(pd.get_dummies(category, prefix=col, dummy_na=True))
        return output

    def calc_nums(self, data, safe_num):
        nums = ['Size', 'SizeOfCode', 'SizeOfHeaders', 'SizeOfImage', 'SizeOfInitializedData',
                'SizeOfUninitializedData', 'FileAlignment', 'ImageBase', 'BaseOfCode', 'BaseOfData',
                'NumberOfSections', 'NumberOfRvaAndSizes', 'Entropy', 'SizeOfOptionalHeader',
                'PointerToSymbolTable', 'NumberOfSymbols']
        pres = [c for c in nums if c in data.columns]
        output = []
        if pres:
            num = data[pres].apply(safe_num)
            if 'Entropy' in num.columns:
                num['Entropy'] = num['Entropy'].clip(0, 8)
            output.append(num)
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

