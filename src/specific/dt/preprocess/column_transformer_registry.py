from src.specific.dt.preprocess.pe_dt_preprocess_map_args import DtPeDataPreprocessMapArgs
from src.specific.dt.preprocess.column_transformer_map_provider import ColumnTransformerMapProvider
from src.specific.dt.preprocess.column_transformer import ColumnTransformer

"""
Provide a ColumnTransformer instance for each supported csv input column name.

In __init__, registry use ColumnTransformerMapProvider to create an Immutable dict between column and transformer.
creates the relevant ColumnTransformer instances, and stores them in transformer_by_column by column name.

The public methods then work only against transformer_by_column:
- get(column_name): return the registered transformer for the column
- contains(column_name): check whether a transformer exists for the column
- columns(): return all registered column names
"""


# Registry that maps column names to ColumnTransformer instances.
class ColumnTransformerRegistry:

    # Args: top identifiers, top imported DLLs and top imported DLLs. Used bit_count is when expanding bitmask fields.
    def __init__(self, provider: ColumnTransformerMapProvider, args: DtPeDataPreprocessMapArgs):
        self.transformer_by_column = dict()
        self.transformer_by_column.update(provider.get_map_transformer_by_column(args))
        self.transformer_by_column.update(provider.get_map_number_by_column())

    # Retrieve transformer for a column. KeyError if column not registered.
    def get(self, column_name: str) -> ColumnTransformer:
        if column_name not in self.transformer_by_column:
            raise KeyError("No transformer registered for column: {}".format(column_name))
        return self.transformer_by_column[column_name]

    def contains(self, column_name: str) -> bool:
        return column_name in self.transformer_by_column

    def columns(self):
        return list(self.transformer_by_column)
