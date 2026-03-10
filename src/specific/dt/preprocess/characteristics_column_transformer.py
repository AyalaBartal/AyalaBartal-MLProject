from typing import Callable, List
import pandas as pd

from src.specific.dt.preprocess.column_transformer import ColumnTransformer


class CharacteristicsColumnTransformer(ColumnTransformer):

    def __init__(self, expand_bits: Callable, safe_num: Callable, prefix: str, bit_count: int):
        # Expands a numeric bitmask column into multiple binary feature columns.
        self.expand_bits = expand_bits
        # safe_num: Convert an arrays of values to numbers and handle bad or missing data.
        self.safe_num = safe_num
        # Name prefix for all new binary feature columns
        self.prefix = prefix
        # Number of bits in bitmask
        self.bit_count = bit_count

    def valid_transform(self, data: pd.DataFrame, column_name: str) -> List[pd.DataFrame]:
        output = []
        output.append(self.get_part_1(data, column_name))
        output.append(self.get_part_2(data, column_name))
        return output

    def get_part_1(self, data, column_name):
        value0 = data[column_name]
        return self.expand_bits(value0, self.bit_count, self.prefix)

    def get_part_2(self, data, column_name):
        value1 = data[column_name]
        value2 = self.safe_num(value1)
        return value2.rename(f"{self.prefix}_raw").to_frame()
