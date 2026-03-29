from typing import Callable, List

import numpy as np
import pandas as pd

from src.common.preprocessor.column_transformer import ColumnTransformer


class FrequencyColumnTransformer(ColumnTransformer):

    def __init__(self, safe_num: Callable):
        # safe_num: Convert an arrays of values to numbers and handle bad or missing data.
        self.safe_num = safe_num

    # What this does:
    # 1. Extract the target column from the input DataFrame.
    # 2. Convert values to numeric using safe_num (invalid values → NaN → replaced with 0).
    # 3. Check if each value is one of the valid PE header sizes: 224 or 240.
    # 4. Convert the boolean result (True/False) to integers (1/0).
    # 5. Convert the Series into a DataFrame with the same column name.
    # 6. Return the DataFrame wrapped in a list to match the transformer interface.
    def valid_transform(self, data: pd.DataFrame, column_name: str) -> List[pd.DataFrame]:
        num = self.safe_num(data[column_name])
        df = num.isin([224, 240]).astype(np.int8).to_frame(name=column_name)
        return [df]

