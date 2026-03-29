from typing import Callable, List
import pandas as pd

from src.common.preprocessor.column_transformer import ColumnTransformer


class RatioColumnTransformer(ColumnTransformer):

    def __init__(self, ratio: Callable, column_a: str, column_b: str):
        # ratio: computes a ratio between two DataFrame, df, columns: a and b.
        self.ratio = ratio
        self.column_a = column_a
        self.column_b = column_b

    """
    What this does:
    1. Check that both required columns (column_a and column_b) exist in the input DataFrame.
    2. Compute the ratio column_a / column_b using the ratio() helper.
       - Values are converted to numeric.
       - Division by zero is avoided.
       - Missing or invalid values become 0.
    3. Convert the resulting Series into a DataFrame.
    4. Return the DataFrame inside a list to match the transformer interface.
    """
    def valid_transform(self, data: pd.DataFrame, column_name: str) -> List[pd.DataFrame]:
        columns_pair = {self.column_a, self.column_b}
        if not columns_pair.issubset(data.columns):
            return []
        ratio_series = self.ratio(data, self.column_a, self.column_b)
        df = ratio_series.to_frame(name=f"{self.column_a}_{self.column_b}_ratio")
        return [df]
