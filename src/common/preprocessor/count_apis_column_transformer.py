from typing import Callable, List
import pandas as pd

from src.common.preprocessor.column_transformer import ColumnTransformer


class CountApisColumnTransformer(ColumnTransformer):

    def __init__(self, parse_listish: Callable, clean_api: Callable):
        # parse_listish: Convert a single  value (number, json, sentence) into a list of simple words.
        self.parse_listish = parse_listish
        # Normalizes API/methods paths.
        self.clean_api = clean_api

    def valid_transform(self, data: pd.DataFrame, column_name: str) -> List[pd.DataFrame]:
        column = data[column_name]
        counts = column.apply(lambda value: len(self.clean_api(self.parse_listish(value))))
        return [counts.to_frame(name=column_name)]
