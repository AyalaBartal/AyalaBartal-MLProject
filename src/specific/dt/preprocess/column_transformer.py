from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List
import pandas as pd

"""
        Transform one DataFrame column into a native Python 1D array.

        Args:
            df: Input pandas DataFrame.
            column_name: Name of the column to transform.

        Returns:
            A native Python 1D array represented as a list.

        Raises:
            ValueError: If the column does not exist or contains invalid data.
            Exception: RuntimeError may raise other exceptions if transformation fails.
        """


class ColumnTransformer(ABC):

    def transform(self, data: pd.DataFrame, column_name: str) -> List[Any]:
        self.validate_transform_args(data, column_name)
        return self.valid_transform(data, column_name)

    def validate_transform_args(self, data, column_name):
        transformer_name = self.__class__.__name__
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame in {} with column {}".format(transformer_name, column_name))
        if column_name not in data.columns:
            raise ValueError("Not found column {} in data at transformer {}".format(column_name, transformer_name))

    @abstractmethod
    def valid_transform(self, df: pd.DataFrame, column_name: str) -> list[Any]: raise RuntimeError
