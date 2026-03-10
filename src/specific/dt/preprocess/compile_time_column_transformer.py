from typing import Callable, List
import pandas as pd

from src.specific.dt.preprocess.column_transformer import ColumnTransformer


class CompileTimeColumnTransformer(ColumnTransformer):

    def __init__(self, parse_tds: Callable, dt_parts: Callable):
        # dt_parts: function that extracts datetime parts
        self.parse_tds = parse_tds
        # to_dt: function that converts series to datetime
        self.dt_parts = dt_parts

    def valid_transform(self, data: pd.DataFrame, column_name: str) -> List[pd.DataFrame]:
        output = []
        dt, an = self.parse_tds(data['TimeDateStamp'])
        output.append(self.dt_parts(dt, 'TDS'))
        output.append(an.rename('timestamp_anomalous').to_frame())
        return output
