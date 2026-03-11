import unittest
import pandas as pd

from src.specific.dt.preprocess.count_apis_column_transformer import CountApisColumnTransformer

"""
Test CountApisColumnTransformer.

CountApisColumnTransformer:
- reads one input column
- converts each cell into a list-like value using parse_listish
- cleans/splits API names using clean_api
- stores the count of cleaned API tokens per row
- returns the result as a list with one DataFrame
"""


class TestCountApisColumnTransformer(unittest.TestCase):

    def test_valid_transform_returns_single_dataframe_with_api_counts(self):
        def parse_listish(value):
            if value == "CreateFile,ReadFile":
                return ["CreateFile", "ReadFile"]
            if value == "CreateFile":
                return ["CreateFile"]
            if value is None:
                return []
            return []

        def clean_api(values):
            # Simulate normalization/splitting of API values.
            return [str(v).lower() for v in values if v]

        data = pd.DataFrame({
            "Apis": ["CreateFile,ReadFile", "CreateFile", None]
        })

        transformer = CountApisColumnTransformer(
            parse_listish=parse_listish,
            clean_api=clean_api,
        )

        result = transformer.valid_transform(data, "Apis")

        self.assertIsInstance(result, list)
        self.assertEqual(1, len(result))

        output_df = result[0]
        self.assertIsInstance(output_df, pd.DataFrame)
        self.assertEqual(["Apis"], list(output_df.columns))
        self.assertEqual([2, 1, 0], output_df["Apis"].tolist())

    def test_valid_transform_preserves_input_index(self):
        def parse_listish(value):
            return value if isinstance(value, list) else []

        def clean_api(values):
            return values

        data = pd.DataFrame(
            {"Apis": [["A", "B"], [], ["X"]]},
            index=[10, 20, 30]
        )

        transformer = CountApisColumnTransformer(
            parse_listish=parse_listish,
            clean_api=clean_api,
        )

        result = transformer.valid_transform(data, "Apis")

        output_df = result[0]
        self.assertEqual([10, 20, 30], list(output_df.index))
        self.assertEqual([2, 0, 1], output_df["Apis"].tolist())

    def test_valid_transform_calls_dependencies_for_each_row(self):
        parse_calls = []
        clean_calls = []

        def parse_listish(value):
            parse_calls.append(value)
            return [value]

        def clean_api(values):
            clean_calls.append(values)
            return values

        data = pd.DataFrame({
            "Apis": ["a", "b", "c"]
        })

        transformer = CountApisColumnTransformer(
            parse_listish=parse_listish,
            clean_api=clean_api,
        )

        result = transformer.valid_transform(data, "Apis")

        self.assertEqual(["a", "b", "c"], parse_calls)
        self.assertEqual([["a"], ["b"], ["c"]], clean_calls)
        self.assertEqual([1, 1, 1], result[0]["Apis"].tolist())