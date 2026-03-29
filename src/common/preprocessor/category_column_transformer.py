from typing import List
import pandas as pd

from src.common.preprocessor.column_transformer import ColumnTransformer


class CategoryColumnTransformer(ColumnTransformer):

    def valid_transform(self, data: pd.DataFrame, column_name: str) -> List[pd.DataFrame]:
        output = []
        category = data[column_name].astype('category')
        output.append(pd.get_dummies(category, prefix=column_name, dummy_na=True))
        return output
