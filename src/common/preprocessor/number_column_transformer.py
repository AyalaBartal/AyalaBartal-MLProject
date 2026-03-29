from typing import Callable, List
import pandas as pd

from src.common.preprocessor.column_transformer import ColumnTransformer


class NumberColumnTransformer(ColumnTransformer):

    def __init__(self, safe_num: Callable):
        # safe_num: Convert an arrays of values to numbers and handle bad or missing data.
        self.safe_num = safe_num

    def valid_transform(self, data: pd.DataFrame, column_name: str) -> List[pd.DataFrame]:
        df = self.safe_num(data[column_name]).to_frame(name=column_name)
        return [df]
