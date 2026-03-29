#!/usr/bin/env python3
"""Preprocess PE CSV for Decision Tree.
- Numeric raw; Entropy clipped
- One-hot categorical
- Bitmasks expanded + raw
- Text/list Top-K counts
- Datetime parts + flags; ratios & structural flags
"""
from src.common.preprocessor.multi_column_transformer import MultiColumnTransformer
from src.common.preprocessor.column_transformer_registry import ColumnTransformerRegistry


class DtPeDataTransformer:

    def __init__(self, registry: ColumnTransformerRegistry):
        # The registry provide transformer per column name.
        self.registry = registry

    # Converts raw input data into a clean numeric feature matrix ready for ML.
    def transform(self, data):
        output = []
        counter = 0
        for column in self.registry.columns():
            transformer = self.registry.get(column)
            t_name = transformer.__class__.__name__
            if isinstance(transformer, MultiColumnTransformer):
                t_name = "{}: {}".format(t_name, transformer.get_transformer_names())
            counter = counter + 1
            print("{}: column={} and transformer={}".format(counter, column, t_name))
            output_list_of_dt = transformer.valid_transform(data, column)
            output.extend(output_list_of_dt)
        return output


