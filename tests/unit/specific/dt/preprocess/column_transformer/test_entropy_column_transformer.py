import unittest
import pandas as pd

from src.common.preprocessor import EntropyColumnTransformer

"""
Test EntropyColumnTransformer.

EntropyColumnTransformer:
- checks whether column 'Entropy' exists in the input DataFrame
- applies safe_num on the existing 'Entropy' column
- returns a list with one DataFrame when the column exists
- returns an empty list when the column does not exist
"""


class TestEntropyColumnTransformer(unittest.TestCase):

    def test_valid_transform_returns_entropy_dataframe_when_column_exists(self):
        def safe_num(series):
            return pd.to_numeric(series, errors="coerce").fillna(0)

        data = pd.DataFrame({
            "Entropy": ["1.5", "2", None]
        })

        transformer = EntropyColumnTransformer(safe_num=safe_num)

        result = transformer.valid_transform(data, "AnyColumnName")

        self.assertIsInstance(result, list)
        self.assertEqual(1, len(result))

        output_df = result[0]
        self.assertIsInstance(output_df, pd.DataFrame)
        self.assertEqual(["Entropy"], list(output_df.columns))
        self.assertEqual([1.5, 2.0, 0.0], output_df["Entropy"].tolist())

    def test_valid_transform_returns_empty_list_when_entropy_column_missing(self):
        def safe_num(series):
            return pd.to_numeric(series, errors="coerce").fillna(0)

        data = pd.DataFrame({
            "OtherColumn": [1, 2, 3]
        })

        transformer = EntropyColumnTransformer(safe_num=safe_num)

        result = transformer.valid_transform(data, "AnyColumnName")

        self.assertEqual([], result)

    def test_valid_transform_preserves_input_index(self):
        def safe_num(series):
            return pd.to_numeric(series, errors="coerce").fillna(0)

        data = pd.DataFrame(
            {"Entropy": ["3.1", None, "4.2"]},
            index=[10, 20, 30]
        )

        transformer = EntropyColumnTransformer(safe_num=safe_num)

        result = transformer.valid_transform(data, "AnyColumnName")

        output_df = result[0]
        self.assertEqual([10, 20, 30], list(output_df.index))
        self.assertEqual([3.1, 0.0, 4.2], output_df["Entropy"].tolist())