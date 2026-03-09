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

        registry = ColumnTransformerRegistry(self.converter, k_dlls, k_apis, k_ident, bit_count)

        output = []
        output.append(self.calc_der(data, safe_num, clean_dll, parse_listish, clean_api))

        # calc_time
        print("columns to transformer: {}".format(registry.columns()))
        for column in registry.columns():
            transformer = registry.get(column)
            output_list_of_dt = transformer.valid_transform(data, column)
            out_class_name = output_list_of_dt.__class__.__name__
            output.extend(output_list_of_dt)

        return output

    def calc_der(self, data, safe_num, clean_dll, parse_listish, clean_api):
        der = pd.DataFrame(index=data.index)

        if 'ImportedDlls' in data.columns:
            der['n_imported_dlls'] = data['ImportedDlls'].apply(lambda v: len(clean_dll(parse_listish(v))))
        if 'ImportedSymbols' in data.columns:
            der['n_imported_symbols'] = data['ImportedSymbols'].apply(lambda v: len(clean_api(parse_listish(v))))
        return der

