from typing import Callable, List

import numpy as np
import pandas as pd

from src.common.preprocessor.column_transformer import ColumnTransformer


class MissingColumnTransformer(ColumnTransformer):

    def valid_transform(self, data: pd.DataFrame, column_name: str) -> List[pd.DataFrame]:
        df = data[[column_name]].isna().astype(np.int8)
        return [df]
