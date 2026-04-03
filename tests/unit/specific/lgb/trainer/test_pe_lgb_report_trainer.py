import unittest
from unittest.mock import create_autospec, Mock, patch
import pandas as pd

from src.specific.lgb.trainer.pe_lgb_report_trainer import LgbPeReportTrainer
from src.specific.lgb.trainer.pe_lgb_data_trainer import LgbPeDataTrainer
from src.specific.lgb.trainer.pe_lgb_model_trainer import LgbPeModelTrainer


class TestLgbPeReportTrainer(unittest.TestCase):

    def setUp(self):
        self.row_selector = create_autospec(LgbPeDataTrainer, instance=True)
        self.matrix_builder = create_autospec(LgbPeModelTrainer, instance=True)

        self.reporter = LgbPeReportTrainer(
            row_selector=self.row_selector,
            matrix_builder=self.matrix_builder,
        )

    def test_init_sets_dependencies(self):
        self.assertIs(self.row_selector, self.reporter.row_selector)
        self.assertIs(self.matrix_builder, self.reporter.matrix_builder)

    def test_init_raises_when_row_selector_is_none(self):
        with self.assertRaises(ValueError) as context:
            LgbPeReportTrainer(None, self.matrix_builder)

        self.assertEqual("row_selector cannot be None", str(context.exception))

    def test_init_raises_when_matrix_builder_is_none(self):
        with self.assertRaises(ValueError) as context:
            LgbPeReportTrainer(self.row_selector, None)

        self.assertEqual("matrix_builder cannot be None", str(context.exception))

    def test_init_raises_when_row_selector_has_wrong_type(self):
        with self.assertRaises(TypeError) as context:
            LgbPeReportTrainer(object(), self.matrix_builder)

        self.assertEqual(
            "row_selector must be of type LgbPeDataTrainer, but got object",
            str(context.exception),
        )

    def test_init_raises_when_matrix_builder_has_wrong_type(self):
        with self.assertRaises(TypeError) as context:
            LgbPeReportTrainer(self.row_selector, object())

        self.assertEqual(
            "matrix_builder must be of type LgbPeModelTrainer, but got object",
            str(context.exception),
        )


if __name__ == "__main__":
    unittest.main()
