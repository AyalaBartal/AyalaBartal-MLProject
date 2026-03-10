from typing import Callable, List
import pandas as pd

from src.specific.dt.preprocess.column_transformer import ColumnTransformer


class IdentifyColumnTransformer(ColumnTransformer):

    def __init__(self, topk: Callable, clean_ident: Callable, k_ident: int):
        # dt_parts: function that extracts datetime parts
        self.topk = topk
        # Normalizes an identifier string and converts it into words
        self.clean_ident = clean_ident
        # Top identifiers
        self.k_ident = k_ident

    def valid_transform(self, data: pd.DataFrame, column_name: str) -> List[pd.DataFrame]:
        out_df = self.topk(data['Identify'], self.clean_ident, self.k_ident, 'id')
        return [out_df]
