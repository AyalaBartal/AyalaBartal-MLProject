import unittest
from unittest.mock import Mock

from src.specific.dt.preprocess import ColumnTransformerRegistry, ColumnTransformer, ColumnTransformerMapProvider


class TestColumnTransformerRegistry(unittest.TestCase):

    def setUp(self):
        self.args = Mock()

        # Mock ColumnTransformer instances. Avoid real subclasses that might change later.
        self.first_seen_transformer = Mock(spec=ColumnTransformer)
        self.imported_dlls_transformer = Mock(spec=ColumnTransformer)
        self.size_transformer = Mock(spec=ColumnTransformer)
        self.image_base_transformer = Mock(spec=ColumnTransformer)

        # Two input dicts returned by provider.
        self.map_transformer_by_column = {
            'FirstSeenDate': self.first_seen_transformer,
            'ImportedDlls': self.imported_dlls_transformer,
        }
        self.map_number_by_column = {
            'Size': self.size_transformer,
            'ImageBase': self.image_base_transformer,
        }

        # Mock provider. Enable to test logic on smaller sample and avoid dependency on real provider code.
        self.provider = Mock(spec=ColumnTransformerMapProvider)
        self.provider.get_map_transformer_by_column.return_value = self.map_transformer_by_column
        self.provider.get_map_number_by_column.return_value = self.map_number_by_column

    def test_get_return_transformer_from_main_map(self):
        registry = ColumnTransformerRegistry(self.provider, self.args)

        result = registry.get('FirstSeenDate')

        self.assertIs(result, self.first_seen_transformer)

    def test_get_return_transformer_from_number_map(self):
        registry = ColumnTransformerRegistry(self.provider, self.args)

        result = registry.get('Size')

        self.assertIs(result, self.size_transformer)

    def test_contains_return_true_for_existing_main_column(self):
        registry = ColumnTransformerRegistry(self.provider, self.args)

        self.assertTrue(registry.contains('ImportedDlls'))

    def test_contains_return_true_for_existing_number_column(self):
        registry = ColumnTransformerRegistry(self.provider, self.args)

        self.assertTrue(registry.contains('ImageBase'))

    def test_contains_return_false_for_missing_column(self):
        registry = ColumnTransformerRegistry(self.provider, self.args)

        self.assertFalse(registry.contains('UnknownColumn'))

    def test_columns_return_all_columns(self):
        registry = ColumnTransformerRegistry(self.provider, self.args)

        result = registry.columns()

        expected = ['FirstSeenDate', 'ImportedDlls', 'Size', 'ImageBase']
        self.assertEqual(result, expected)

    def test_get_raise_key_error_for_missing_column(self):
        registry = ColumnTransformerRegistry(self.provider, self.args)

        with self.assertRaises(KeyError) as cm:
            registry.get('UnknownColumn')

        expected = "'No transformer registered for column: UnknownColumn'"
        self.assertEqual(str(cm.exception),expected)

    def test_init_call_provider_methods(self):
        ColumnTransformerRegistry(self.provider, self.args)

        self.provider.get_map_transformer_by_column.assert_called_once_with(self.args)
        self.provider.get_map_number_by_column.assert_called_once_with()
