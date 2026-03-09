import pandas as pd
import numpy as np
import re


class DtPePreprocessMapper:

    def __init__(self, args, transformer):
        self.args = args
        self.transformer = transformer

    def map(self, input_data):
        args = self.args
        out_array = self.transformer.transform(input_data)
        # Create DataFrame from out_array if it exits or else: return empty DataFrame with the correct row count.
        output_data = pd.concat(out_array, axis=1) if out_array else pd.DataFrame(index=input_data.index)
        # Remove infinite and missing values with zeros.
        output_data = output_data.replace([np.inf, -np.inf], 0).fillna(0)
        # Sanitizes column names.
        output_data.columns = [re.sub(r"[^0-9A-Za-z_]+", "_", str(c)) for c in output_data.columns]
        # Add label_col column to output
        if args.label_col and args.label_col in input_data.columns:
            output_data[args.label_col] = input_data[args.label_col].values
        return output_data
