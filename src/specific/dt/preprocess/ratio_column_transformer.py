from typing import Callable, List
import pandas as pd

from src.specific.dt.preprocess.column_transformer import ColumnTransformer


class RatioColumnTransformer(ColumnTransformer):

    def __init__(self, ratio: Callable, column_a: str, column_b: str):
        # ratio: computes a ratio between two DataFrame, df, columns: a and b.
        self.ratio = ratio
        self.column_a = column_a
        self.column_b = column_b

    def valid_transform(self, data: pd.DataFrame, column_name: str) -> List[pd.DataFrame]:
        columns_pair = {self.column_a, self.column_b}
        if columns_pair.issubset(data.columns):
            return self.ratio(data, self.column_a, self.column_b)
        return []
