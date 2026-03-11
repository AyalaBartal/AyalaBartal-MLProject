import unittest
import pandas as pd
from pandas.testing import assert_frame_equal

from src.specific.dt.preprocess import ColumnTransformer, MultiColumnTransformer


# Small test helper that returns pre-defined frames.
class FirstDummyColumnTransformer(ColumnTransformer):

    def __init__(self, frames):
        self.frames = frames
        self.calls = []

    def valid_transform(self, input_data, input_column_name):
        self.calls.append((input_data.copy(), input_column_name))
        return self.frames


class SecondDummyColumnTransformer(ColumnTransformer):
    def valid_transform(self, input_data, input_column_name):
        return []


class TestMultiColumnTransformer(unittest.TestCase):

    def test_init_raises_when_transformer_by_name_is_not_dict(self):
        with self.assertRaises(TypeError):
            invalid = ["not", "a", "dict"]
            MultiColumnTransformer(invalid)

    def test_init_raises_when_key_is_not_str(self):
        child = FirstDummyColumnTransformer([])
        invalid = {123: child}
        with self.assertRaises(TypeError):
            MultiColumnTransformer(invalid)

    def test_init_raises_when_value_is_not_column_transformer(self):
        with self.assertRaises(TypeError):
            MultiColumnTransformer({"x_{}": object()})

    def test_init_stores_transformer_by_name(self):
        t1 = FirstDummyColumnTransformer([])
        t2 = SecondDummyColumnTransformer()

        transformer = MultiColumnTransformer({
            "first_{}": t1,
            "second_{}": t2,
        })

        self.assertEqual(["first_{}", "second_{}"], list(transformer.transformer_by_name.keys()))

    def test_valid_transform_calls_each_child_transformer(self):
        input_data = pd.DataFrame({"A": [1, 2]})
        t1 = FirstDummyColumnTransformer([pd.DataFrame({"x": [10, 20]})])
        t2 = FirstDummyColumnTransformer([pd.DataFrame({"y": [30, 40]})])

        transformer = MultiColumnTransformer({
            "left_{}": t1,
            "right_{}": t2,
        })

        result = transformer.valid_transform(input_data, "A")

        self.assertEqual(1, len(t1.calls))
        self.assertEqual(1, len(t2.calls))
        self.assertEqual("A", t1.calls[0][1])
        self.assertEqual("A", t2.calls[0][1])
        self.assertEqual(2, len(result))

    def test_valid_transform_renames_columns_with_target_template(self):
        input_data = pd.DataFrame({"A": [1, 2]})
        child = FirstDummyColumnTransformer([
            pd.DataFrame({
                "mean": [10, 20],
                "max": [30, 40],
            })
        ])

        transformer = MultiColumnTransformer({
            "feature_{}": child
        })

        result = transformer.valid_transform(input_data, "A")

        expected = pd.DataFrame({
            "feature_mean": [10, 20],
            "feature_max": [30, 40],
        })

        self.assertEqual(1, len(result))
        assert_frame_equal(expected, result[0])

    def test_valid_transform_combines_frames_from_multiple_children(self):
        input_data = pd.DataFrame({"A": [1, 2]})
        t1 = FirstDummyColumnTransformer([
            pd.DataFrame({"x": [1, 2]}),
            pd.DataFrame({"y": [3, 4]}),
        ])
        t2 = FirstDummyColumnTransformer([
            pd.DataFrame({"z": [5, 6]})
        ])

        transformer = MultiColumnTransformer({
            "one_{}": t1,
            "two_{}": t2,
        })

        result = transformer.valid_transform(input_data, "A")

        expected_0 = pd.DataFrame({"one_x": [1, 2]})
        expected_1 = pd.DataFrame({"one_y": [3, 4]})
        expected_2 = pd.DataFrame({"two_z": [5, 6]})

        self.assertEqual(3, len(result))
        assert_frame_equal(expected_0, result[0])
        assert_frame_equal(expected_1, result[1])
        assert_frame_equal(expected_2, result[2])

    def test_rename_frames_returns_empty_list_for_empty_input(self):
        transformer = MultiColumnTransformer({})

        result = transformer.rename_frames([], "prefix_{}")

        self.assertEqual([], result)

    def test_rename_frames_preserves_original_frame(self):
        transformer = MultiColumnTransformer({})
        original = pd.DataFrame({"a": [1, 2]})

        result = transformer.rename_frames([original], "renamed_{}")

        expected_result = pd.DataFrame({"renamed_a": [1, 2]})
        expected_original = pd.DataFrame({"a": [1, 2]})

        assert_frame_equal(expected_result, result[0])
        assert_frame_equal(expected_original, original)

    def test_rename_frames_raises_when_frame_is_not_dataframe(self):
        transformer = MultiColumnTransformer({})

        with self.assertRaises(TypeError):
            transformer.rename_frames([["not a dataframe"]], "x_{}")

    def test_get_transformer_names_returns_sorted_class_names(self):
        transformer = MultiColumnTransformer({
            "b_{}": FirstDummyColumnTransformer([]),
            "a_{}": SecondDummyColumnTransformer(),
        })

        result = transformer.get_transformer_names()

        expected = ["FirstDummyColumnTransformer", "SecondDummyColumnTransformer"]

        self.assertEqual(expected, result)
