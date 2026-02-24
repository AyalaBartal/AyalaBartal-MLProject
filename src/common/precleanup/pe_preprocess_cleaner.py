
"""
Pipeline → Stage → Step
# pipline: ML for malware in PE using algorithm DT
# Stage: Preprocess
# Step: clean csv from irrelevant columns and invalid rows
"""
import pandas as pd


def is_even(x):
    return x.notna() & (x % 1 == 0)


class PePreprocessCleaner:

    def __init__(self, columns_provider):
        self.columns_provider = columns_provider

    def clean(self, data):
        data['Identify'] = data['Identify'].fillna('unknown')
        data = data.dropna()

        # keep only rows where all specified columns are valid integers
        integer_columns = self.columns_provider.get_non_negative_integer_headers()
        data = data[data[integer_columns].apply(pd.to_numeric, errors='coerce').pipe(is_even).all(axis=1)]

        # Filter rows where 'Entropy' is between 0.0 and 20.0
        float_column = self.columns_provider.get_positive_float_headers()[0]
        data = data[data[float_column].between(0.0, 20.0)]

        # keep only rows where all specified columns are valid dates
        date_columns = self.columns_provider.get_date_headers()
        data = data[data[date_columns].apply(lambda col: pd.to_datetime(col, errors='coerce').notna()).all(axis=1)]

        # Keep only rows where all specified columns have valid not empty text
        text_columns = self.columns_provider.get_text_headers()
        data = data[data[text_columns].apply(lambda s: s.notna() & s.astype(str).str.strip().ne('')).all(axis=1)]

        # Keep only rows where a  column contains valid binary values (0 or 1)
        data = data[pd.to_numeric(data['Label'], errors='coerce').isin([0, 1])]

        # Remove all rows that contain any missing value (NaN, None)
        data = data.dropna()

        return data
