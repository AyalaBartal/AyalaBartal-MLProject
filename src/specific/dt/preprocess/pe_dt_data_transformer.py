#!/usr/bin/env python3
"""Preprocess PE CSV for Decision Tree.
- Numeric raw; Entropy clipped
- One-hot categoricals
- Bitmasks expanded + raw
- Text/list Top-K counts
- Datetime parts + flags; ratios & structural flags
"""
import re
import numpy as np
import pandas as pd


class DtPeDataTransformer:

    def __init__(self, converter):
        self.converter = converter

    def transform(self, data, k_ident=100, k_dlls=100, k_apis=200, bit_count=16):
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

        parts = []
        parts.extend(self.calc_nums(data, safe_num))
        parts.append(self.calc_der(data, ratio, safe_num, clean_dll, parse_listish, clean_api))
        parts2 = self.calc_characteristics(data, bit_count, expand_bits, safe_num)
        parts.extend(parts2)

        if 'Identify' in data.columns:
            parts.append(topk(data['Identify'], clean_ident, k_ident, 'id'))

        if 'ImportedDlls' in data.columns:
            parts.append(topk(data['ImportedDlls'], lambda v: clean_dll(parse_listish(v)), k_dlls, 'dll'))
        if 'ImportedSymbols' in data.columns:
            parts.append(topk(data['ImportedSymbols'], lambda v: clean_api(parse_listish(v)), k_apis, 'api'))

        if 'FirstSeenDate' in data.columns:
            dt = to_dt(data['FirstSeenDate'])
            parts.append(dt_parts(dt, 'FirstSeen'))
            parts.append(dt.isna().astype(np.int8).rename('FirstSeen_missing').to_frame())
        if 'TimeDateStamp' in data.columns:
            dt, an = parse_tds(data['TimeDateStamp'])
            parts.append(dt_parts(dt, 'TDS'))
            parts.append(an.rename('timestamp_anomalous').to_frame())

        X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=data.index)
        X = X.replace([np.inf, -np.inf], 0).fillna(0)
        X.columns = [re.sub(r"[^0-9A-Za-z_]+", "_", str(c)) for c in X.columns]
        return X

    def calc_characteristics(self, data, bit_count, expand_bits, safe_num):
        parts2 = []
        for col, p in [('Characteristics', 'char'), ('DllCharacteristics', 'dllc')]:
            if col in data.columns:
                if expand_bits:
                    value0 = data[col]
                    part1 = expand_bits(value0, bit_count, p)
                    parts2.append(part1)
                value1 = data[col]
                value2 = safe_num(value1)
                value3 = value2.rename(f"{p}_raw").to_frame()
                parts2.append(value3)
        for col in ['Machine', 'PE_TYPE']:
            if col in data.columns:
                category = data[col].astype('category')
                parts2.append(pd.get_dummies(category, prefix=col, dummy_na=True))
        return parts2

    def calc_nums(self, data, safe_num):
        nums = ['Size', 'SizeOfCode', 'SizeOfHeaders', 'SizeOfImage', 'SizeOfInitializedData',
                'SizeOfUninitializedData', 'FileAlignment', 'ImageBase', 'BaseOfCode', 'BaseOfData',
                'NumberOfSections', 'NumberOfRvaAndSizes', 'Entropy', 'SizeOfOptionalHeader',
                'PointerToSymbolTable', 'NumberOfSymbols']
        pres = [c for c in nums if c in data.columns]
        parts2 = []
        if pres:
            num = data[pres].apply(safe_num)
            if 'Entropy' in num.columns:
                num['Entropy'] = num['Entropy'].clip(0, 8)
            parts2.append(num)
        return parts2

    def calc_der(self, data, ratio, safe_num, clean_dll, parse_listish, clean_api):
        der = pd.DataFrame(index=data.index)
        if {'SizeOfCode', 'SizeOfImage'}.issubset(data.columns):
            der['ratio_Code_Image'] = ratio(data, 'SizeOfCode', 'SizeOfImage')
        if {'SizeOfInitializedData', 'Size'}.issubset(data.columns):
            der['ratio_InitData_Size'] = ratio(data, 'SizeOfInitializedData', 'Size')
        if {'SizeOfHeaders', 'Size'}.issubset(data.columns):
            der['ratio_Headers_Size'] = ratio(data, 'SizeOfHeaders', 'Size')
        if 'BaseOfData' in data.columns: der['BaseOfData_missing'] = data['BaseOfData'].isna().astype(np.int8)
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


