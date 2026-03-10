import unittest

from src.specific.dt.preprocess import DtPePreprocessorProvider, DtPePreprocessMapper, DtPeDataTransformer


class TestDtPePreprocessorProvider(unittest.TestCase):

    def test_get_mapper_returns_mapper(self):
        actual = DtPePreprocessorProvider.get_mapper()

        self.assertIsNotNone(actual)
        self.assertIsInstance(actual, DtPePreprocessMapper)

    def test_get_transformer_returns_transformer(self):
        actual = DtPePreprocessorProvider.get_transformer()

        self.assertIsNotNone(actual)
        self.assertIsInstance(actual, DtPeDataTransformer)
