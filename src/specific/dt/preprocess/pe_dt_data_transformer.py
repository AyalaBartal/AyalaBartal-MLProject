#!/usr/bin/env python3
"""Preprocess PE CSV for Decision Tree.
- Numeric raw; Entropy clipped
- One-hot categorical
- Bitmasks expanded + raw
- Text/list Top-K counts
- Datetime parts + flags; ratios & structural flags
"""

from src.specific.dt.preprocess.column_transformer_registry import ColumnTransformerRegistry


class DtPeDataTransformer:

    def __init__(self, converter):
        self.converter = converter

    # Converts raw input data into a clean numeric feature matrix ready for ML.
    # Args: top identifiers, top imported DLLs and top imported DLLs. Used bit_count is when expanding bitmask fields.
    def transform(self, data, k_ident=100, k_dlls=100, k_apis=200, bit_count=16):
        registry = ColumnTransformerRegistry(self.converter, k_dlls, k_apis, k_ident, bit_count)
        output = []
        for column in registry.columns():
            transformer = registry.get(column)
            output_list_of_dt = transformer.valid_transform(data, column)
            output.extend(output_list_of_dt)

        return output


