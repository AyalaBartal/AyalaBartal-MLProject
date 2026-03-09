from __future__ import annotations

from typing import Dict, List
import pandas as pd

from src.specific.dt.preprocess.column_transformer import ColumnTransformer

"""
Map one input column into multiple output columns using Dict transformer_by_name.
transformer_by_name:
    key   -> target output column name
    value -> ColumnTransformer that generates the data for that column
"""


class MultiColumnTransformer(ColumnTransformer):

    def __init__(self, transformer_by_name: Dict[str, ColumnTransformer]) -> None:
        self.validate_transformer_by_name(transformer_by_name)
        self.transformer_by_name = transformer_by_name

    def validate_transformer_by_name(self, transformer_by_name):
        if not isinstance(transformer_by_name, dict):
            raise TypeError("transformer_by_name must be a dict[str, ColumnTransformer]")

        for target_column_name, transformer in transformer_by_name.items():
            if not isinstance(target_column_name, str):
                raise TypeError("All keys in transformer_by_name must be str")
            if not isinstance(transformer, ColumnTransformer):
                raise TypeError("Value for key '{}' must be a ColumnTransformer".format(target_column_name))

    def valid_transform(self, input_data: pd.DataFrame, input_column_name: str) -> List[pd.DataFrame]:
        out_all = []
        for target_column_name, transformer in self.transformer_by_name.items():
            before_rename_frames = transformer.transform(input_data, input_column_name)
            after_rename_frames = self.rename_frames(before_rename_frames, target_column_name)
            out_all.extend(after_rename_frames)
        return out_all

    def get_transformer_names(self) -> List[str]:
        values = list(self.transformer_by_name.values())
        names = [v.__class__.__name__ for v in values]
        return sorted(names)

    def rename_frames(self, before_rename_frames: List[pd.DataFrame], template_column_name: str) -> List[pd.DataFrame]:
        after_rename_frames = []
        for frame in before_rename_frames:
            self.validate_frame(frame, template_column_name)
            renamed_frame = frame.copy()
            renamed_frame.columns = [template_column_name.format(col) for col in renamed_frame.columns]
            after_rename_frames.append(renamed_frame)
        return after_rename_frames

    def validate_frame(self, frame, target_column_name):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Transformer for '{}' returned non-DataFrame result".format(target_column_name))
