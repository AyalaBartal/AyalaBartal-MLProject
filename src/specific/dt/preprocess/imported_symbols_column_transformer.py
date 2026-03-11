from typing import Callable, List
import pandas as pd

from src.specific.dt.preprocess.column_transformer import ColumnTransformer


class ImportedSymbolsColumnTransformer(ColumnTransformer):

    def __init__(self, parse_listish: Callable, clean_api: Callable, topk: Callable, k_apis: int):
        # parse_listish: Convert a single  value (number, json, sentence) into a list of simple words.
        self.parse_listish = parse_listish
        # Normalizes API/methods paths.
        self.clean_api = clean_api
        # topk: converts a text column into numeric ML features using the top K most common words.
        self.topk = topk
        # Number of top APIs/symbols to select
        self.k_top = k_apis

    def valid_transform(self, data: pd.DataFrame, column_name: str) -> List[pd.DataFrame]:
        # topk(data, mapper, k_top, output_columns_prefix)
        out_df = self.topk(data['ImportedSymbols'], self.process_one_cell_value, self.k_top, 'api')
        return [out_df]

    def process_one_cell_value(self, value):
        return self.clean_api(self.parse_listish(value))
