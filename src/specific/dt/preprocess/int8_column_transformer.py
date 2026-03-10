from typing import Callable, List

import numpy as np
import pandas as pd

from src.specific.dt.preprocess.column_transformer import ColumnTransformer


class Int8ColumnTransformer(ColumnTransformer):

    def __init__(self, safe_num: Callable):
        self.safe_num = safe_num

    def valid_transform(self, data: pd.DataFrame, column_name: str) -> List[pd.DataFrame]:
        column_data = data[column_name]
        positive_numbers = self.safe_num(column_data) > 0
        df = positive_numbers.astype(np.int8).to_frame(name=column_name)
        return [df]
