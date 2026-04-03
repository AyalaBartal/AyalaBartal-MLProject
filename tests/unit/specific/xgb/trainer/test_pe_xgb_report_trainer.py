import unittest
from unittest.mock import create_autospec, Mock, patch
import pandas as pd

from src.specific.xgb.trainer.pe_xgb_report_trainer import XgbPeReportTrainer
from src.specific.xgb.trainer.pe_xgb_data_trainer import XgbPeDataTrainer
from src.specific.xgb.trainer.pe_xgb_model_trainer import XgbPeModelTrainer


class TestXgbPeReportTrainer(unittest.TestCase):

    def setUp(self):
        self.row_selector = create_autospec(XgbPeDataTrainer, instance=True)
        self.matrix_builder = create_autospec(XgbPeModelTrainer, instance=True)

        self.reporter = XgbPeReportTrainer(
            row_selector=self.row_selector,
            matrix_builder=self.matrix_builder,
        )

    def test_init_sets_dependencies(self):
        self.assertIs(self.row_selector, self.reporter.row_selector)
        self.assertIs(self.matrix_builder, self.reporter.matrix_builder)

    def test_init_raises_when_row_selector_is_none(self):
        with self.assertRaises(ValueError) as context:
            XgbPeReportTrainer(None, self.matrix_builder)

        self.assertEqual("row_selector cannot be None", str(context.exception))

    def test_init_raises_when_matrix_builder_is_none(self):
        with self.assertRaises(ValueError) as context:
            XgbPeReportTrainer(self.row_selector, None)

        self.assertEqual("matrix_builder cannot be None", str(context.exception))

    def test_init_raises_when_row_selector_has_wrong_type(self):
        with self.assertRaises(TypeError) as context:
            XgbPeReportTrainer(object(), self.matrix_builder)

        self.assertEqual(
            "row_selector must be of type XgbPeDataTrainer, but got object",
            str(context.exception),
        )

    def test_init_raises_when_matrix_builder_has_wrong_type(self):
        with self.assertRaises(TypeError) as context:
            XgbPeReportTrainer(self.row_selector, object())

        self.assertEqual(
            "matrix_builder must be of type XgbPeModelTrainer, but got object",
            str(context.exception),
        )


if __name__ == "__main__":
    unittest.main()

