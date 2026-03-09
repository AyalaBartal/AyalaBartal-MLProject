from typing import Callable, List

import numpy as np
import pandas as pd

from src.specific.dt.preprocess.column_transformer import ColumnTransformer


class FrequencyColumnTransformer(ColumnTransformer):

    def __init__(self, safe_num: Callable):
        # safe_num: Convert an arrays of values to numbers and handle bad or missing data.
        self.safe_num = safe_num

    def valid_transform(self, data: pd.DataFrame, column_name: str) -> List[pd.DataFrame]:
        num = self.safe_num(data['SizeOfOptionalHeader'])
        return num.isin([224, 240]).astype(np.int8)
