import pandas as pd
import numpy as np
import re

from src.common.preprocessor.pe_dt_preprocess_map_args import DtPeDataPreprocessMapArgs
from src.common.preprocessor.pe_dt_data_transformer import DtPeDataTransformer


class DtPePreprocessMapper:

    def __init__(self, args: DtPeDataPreprocessMapArgs, transformer: DtPeDataTransformer):
        self.args = args
        self.transformer = transformer

    def map(self, input_data):
        out_array = self.transformer.transform(input_data)
        # Create DataFrame from out_array if it exits or else: return empty DataFrame with the correct row count.
        output_data = pd.concat(out_array, axis=1) if out_array else pd.DataFrame(index=input_data.index)
        # Remove infinite and missing values with zeros.
        output_data = output_data.replace([np.inf, -np.inf], 0).fillna(0)
        # Sanitizes column names.
        output_data.columns = [re.sub(r"[^0-9A-Za-z_]+", "_", str(c)) for c in output_data.columns]
        # Add label_col column to output
        out_col = self.args.label_col
        input_columns = list(input_data.columns)
        if out_col and out_col in input_columns:
            output_data[out_col] = input_data[out_col].values
        # Sort columns by alphabet order
        output_data = output_data.sort_index(axis=1)
        return output_data
