import unittest
from unittest.mock import Mock

from src.specific.dt.preprocess import DtPeDataTransformer


class TestDtDtPeDataTransformer(unittest.TestCase):

    def test_transform_empty_registry_returns_empty_list(self):
        registry = Mock()
        registry.columns.return_value = []

        transformer = DtPeDataTransformer(registry)

        data = {"x": [1, 2]}
        actual = transformer.transform(data)

        self.assertEqual(actual, [])
        registry.columns.assert_called_once_with()
        registry.get.assert_not_called()

    def test_transform_single_column_single_output(self):
        registry = Mock()
        column_transformer = Mock()

        registry.columns.return_value = ["col_a"]
        registry.get.return_value = column_transformer
        column_transformer.valid_transform.return_value = ["df_a"]

        transformer = DtPeDataTransformer(registry)

        data = {"col_a": [1, 2]}
        actual = transformer.transform(data)

        self.assertEqual(actual, ["df_a"])
        registry.columns.assert_called_once_with()
        registry.get.assert_called_once_with("col_a")
        column_transformer.valid_transform.assert_called_once_with(data, "col_a")

    def test_transform_single_column_multiple_outputs(self):
        registry = Mock()
        column_transformer = Mock()

        registry.columns.return_value = ["col_a"]
        registry.get.return_value = column_transformer
        column_transformer.valid_transform.return_value = ["df_a1", "df_a2"]

        transformer = DtPeDataTransformer(registry)

        data = {"col_a": [1, 2]}
        actual = transformer.transform(data)

        self.assertEqual(actual, ["df_a1", "df_a2"])

    def test_transform_multiple_columns_flattens_all_outputs_in_order(self):
        registry = Mock()

        transformer_a = Mock()
        transformer_b = Mock()

        registry.columns.return_value = ["col_a", "col_b"]

        def get_transformer(column):
            if column == "col_a":
                return transformer_a
            if column == "col_b":
                return transformer_b
            raise KeyError(column)

        registry.get.side_effect = get_transformer

        transformer_a.valid_transform.return_value = ["a1", "a2"]
        transformer_b.valid_transform.return_value = ["b1"]

        transformer = DtPeDataTransformer(registry)

        data = {"col_a": [1], "col_b": [2]}
        actual = transformer.transform(data)

        self.assertEqual(actual, ["a1", "a2", "b1"])
        self.assertEqual(registry.get.call_args_list[0].args[0], "col_a")
        self.assertEqual(registry.get.call_args_list[1].args[0], "col_b")
        transformer_a.valid_transform.assert_called_once_with(data, "col_a")
        transformer_b.valid_transform.assert_called_once_with(data, "col_b")

    def test_transform_skips_column_when_transformer_returns_empty_list(self):
        registry = Mock()

        transformer_a = Mock()
        transformer_b = Mock()

        registry.columns.return_value = ["col_a", "col_b"]

        def get_transformer(column):
            if column == "col_a":
                return transformer_a
            if column == "col_b":
                return transformer_b
            raise KeyError(column)

        registry.get.side_effect = get_transformer

        transformer_a.valid_transform.return_value = []
        transformer_b.valid_transform.return_value = ["b1"]

        transformer = DtPeDataTransformer(registry)

        data = {"col_a": [1], "col_b": [2]}
        actual = transformer.transform(data)

        self.assertEqual(actual, ["b1"])

    def test_transform_preserves_order_of_registry_columns(self):
        registry = Mock()

        transformer_a = Mock()
        transformer_b = Mock()
        transformer_c = Mock()

        registry.columns.return_value = ["col_b", "col_c", "col_a"]

        def get_transformer(column):
            mapping = {
                "col_a": transformer_a,
                "col_b": transformer_b,
                "col_c": transformer_c,
            }
            return mapping[column]

        registry.get.side_effect = get_transformer

        transformer_a.valid_transform.return_value = ["a"]
        transformer_b.valid_transform.return_value = ["b"]
        transformer_c.valid_transform.return_value = ["c"]

        transformer = DtPeDataTransformer(registry)

        data = {"col_a": [1], "col_b": [2], "col_c": [3]}
        actual = transformer.transform(data)

        self.assertEqual(actual, ["b", "c", "a"])
