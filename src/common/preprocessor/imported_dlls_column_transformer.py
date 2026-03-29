from typing import Callable, List
import pandas as pd

from src.common.preprocessor.column_transformer import ColumnTransformer


class ImportedDllsColumnTransformer(ColumnTransformer):

    def __init__(self, parse_listish: Callable, clean_dll: Callable, topk: Callable, k_dlls: int):
        # parse_listish: Convert a single  value (number, json, sentence) into a list of simple words.
        self.parse_listish = parse_listish
        # Normalizes DLL names from a sequence (ts) to a list of DLL names without paths and without the .dll extension.
        self.clean_dll = clean_dll
        # topk: converts a text column into numeric ML features using the top K most common words.
        self.topk = topk
        # Number of top dlls to select
        self.k_top = k_dlls

    def valid_transform(self, data: pd.DataFrame, column_name: str) -> List[pd.DataFrame]:
        # topk(data, mapper, k_top, output_columns_prefix)
        out_df = self.topk(data['ImportedDlls'], self.process_one_cell_value, self.k_top, 'dll')
        return [out_df]

    def process_one_cell_value(self, value):
        return self.clean_dll(self.parse_listish(value))
