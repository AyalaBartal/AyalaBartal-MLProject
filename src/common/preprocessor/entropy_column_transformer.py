from typing import Callable, List
import pandas as pd

from src.common.preprocessor.column_transformer import ColumnTransformer


class EntropyColumnTransformer(ColumnTransformer):

    def __init__(self, safe_num: Callable):
        # safe_num: Convert an arrays of values to numbers and handle bad or missing data.
        self.safe_num = safe_num

    def valid_transform(self, data: pd.DataFrame, column_name: str) -> List[pd.DataFrame]:
        nums = ['Entropy']
        pres = [c for c in nums if c in data.columns]
        output = []
        if pres:
            num = data[pres].apply(self.safe_num)
            output.append(num)
        return output
