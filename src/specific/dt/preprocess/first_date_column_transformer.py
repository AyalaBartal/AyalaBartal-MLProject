from typing import Any, Callable, List
import pandas as pd
import numpy as np

from src.specific.dt.preprocess.column_transformer import ColumnTransformer


class FirstDateColumnTransformer(ColumnTransformer):


    def __init__(self, dt_parts: Callable, to_dt: Callable):
        # dt_parts: convert timestamp into columns (year, month, dow)
        self.dt_parts = dt_parts
        # to_dt: converts a value (or column) into a datetime using pandas.
        self.to_dt = to_dt

    def valid_transform(self, data: pd.DataFrame, column_name: str) -> List[Any]:
        output = []
        dt = self.to_dt(data['FirstSeenDate'])
        output.append(self.dt_parts(dt, 'FirstSeen'))
        output.append(dt.isna().astype(np.int8).rename('FirstSeen_missing').to_frame())
        return output
