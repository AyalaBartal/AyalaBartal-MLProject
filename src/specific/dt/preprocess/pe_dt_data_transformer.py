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

    def __init__(self, registry):
        # The registry provide transformer per column name.
        self.registry = registry

    # Converts raw input data into a clean numeric feature matrix ready for ML.
    def transform(self, data):
        output = []
        for column in self.registry.columns():
            transformer = self.registry.get(column)
            output_list_of_dt = transformer.valid_transform(data, column)
            output.extend(output_list_of_dt)
        return output


