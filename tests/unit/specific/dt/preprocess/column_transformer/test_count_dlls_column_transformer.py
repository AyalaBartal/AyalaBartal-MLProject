import unittest
import pandas as pd

from src.common.preprocessor import CountDllsColumnTransformer

"""
Test CountDllsColumnTransformer.

CountDllsColumnTransformer:
- reads a single column
- converts each cell value to a list using parse_listish
- normalizes DLL names using clean_dll
- outputs the count of DLLs per row
- returns a list containing one DataFrame
"""


class TestCountDllsColumnTransformer(unittest.TestCase):

    def test_valid_transform_counts_dlls(self):
        def parse_listish(value):
            if value == "kernel32.dll,user32.dll":
                return ["kernel32.dll", "user32.dll"]
            if value == "kernel32.dll":
                return ["kernel32.dll"]
            return []

        def clean_dll(values):
            # Simulate cleaning: remove extension
            return [v.replace(".dll", "") for v in values]

        data = pd.DataFrame({
            "Dlls": [
                "kernel32.dll,user32.dll",
                "kernel32.dll",
                None
            ]
        })

        transformer = CountDllsColumnTransformer(
            parse_listish=parse_listish,
            clean_dll=clean_dll
        )

        result = transformer.valid_transform(data, "Dlls")

        self.assertIsInstance(result, list)
        self.assertEqual(1, len(result))

        output_df = result[0]

        self.assertIsInstance(output_df, pd.DataFrame)
        self.assertEqual(["Dlls"], list(output_df.columns))
        self.assertEqual([2, 1, 0], output_df["Dlls"].tolist())

    def test_valid_transform_preserves_index(self):
        def parse_listish(value):
            return value if isinstance(value, list) else []

        def clean_dll(values):
            return values

        data = pd.DataFrame(
            {"Dlls": [["a", "b"], [], ["x"]]},
            index=[10, 20, 30]
        )

        transformer = CountDllsColumnTransformer(parse_listish, clean_dll)

        result = transformer.valid_transform(data, "Dlls")

        output_df = result[0]

        self.assertEqual([10, 20, 30], list(output_df.index))
        self.assertEqual([2, 0, 1], output_df["Dlls"].tolist())
