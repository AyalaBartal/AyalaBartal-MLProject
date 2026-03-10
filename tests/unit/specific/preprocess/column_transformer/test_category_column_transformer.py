import unittest
import pandas as pd
from pandas.testing import assert_frame_equal

from src.specific.dt.preprocess import CategoryColumnTransformer


class TestCategoryColumnTransformer(unittest.TestCase):

    def setUp(self):
        self.transformer = CategoryColumnTransformer()

    def test_valid_transform_returns_list_with_dataframe(self):
        df = pd.DataFrame({"color": ["red", "blue", "red"]})

        result = self.transformer.valid_transform(df, "color")

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], pd.DataFrame)

    def test_valid_transform_creates_correct_dummy_columns(self):
        df = pd.DataFrame({"color": ["red", "blue", "red"]})

        result = self.transformer.valid_transform(df, "color")[0]

        expected = pd.get_dummies(
            df["color"].astype("category"),
            prefix="color",
            dummy_na=True
        )

        assert_frame_equal(result, expected)

    def test_valid_transform_handles_nan_values(self):
        df = pd.DataFrame({"color": ["red", None, "blue"]})

        result = self.transformer.valid_transform(df, "color")[0]

        expected = pd.get_dummies(
            df["color"].astype("category"),
            prefix="color",
            dummy_na=True
        )

        assert_frame_equal(result, expected)

    def test_valid_transform_preserves_index(self):
        df = pd.DataFrame(
            {"color": ["red", "blue"]},
            index=[10, 11]
        )

        result = self.transformer.valid_transform(df, "color")[0]

        self.assertTrue(result.index.equals(df.index))

    def test_valid_transform_multiple_categories(self):
        df = pd.DataFrame({"color": ["red", "blue", "green"]})

        result = self.transformer.valid_transform(df, "color")[0]

        self.assertIn("color_red", result.columns)
        self.assertIn("color_blue", result.columns)
        self.assertIn("color_green", result.columns)
        self.assertIn("color_nan", result.columns)
