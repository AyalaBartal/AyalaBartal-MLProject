from typing import Callable, List

import numpy as np
import pandas as pd

from src.specific.dt.preprocess.column_transformer import ColumnTransformer


class MissingColumnTransformer(ColumnTransformer):

    def valid_transform(self, data: pd.DataFrame, column_name: str) -> List[pd.DataFrame]:
        return data[column_name].isna().astype(np.int8)
