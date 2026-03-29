from typing import Callable, List
import pandas as pd

from src.common.preprocessor.column_transformer import ColumnTransformer


class CountDllsColumnTransformer(ColumnTransformer):

    def __init__(self, parse_listish: Callable, clean_dll: Callable):
        # parse_listish: Convert a single  value (number, json, sentence) into a list of simple words.
        self.parse_listish = parse_listish
        # Normalizes DLL names from a sequence (ts) to a list of DLL names without paths and without the .dll extension.
        self.clean_dll = clean_dll

    def valid_transform(self, data: pd.DataFrame, column_name: str) -> List[pd.DataFrame]:
        column = data[column_name]
        counts = column.apply(lambda value: len(self.clean_dll(self.parse_listish(value))))
        return [counts.to_frame(name=column_name)]
