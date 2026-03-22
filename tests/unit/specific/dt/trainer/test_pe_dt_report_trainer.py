import unittest
from unittest.mock import MagicMock

from src.specific.dt.trainer.pe_dt_report_trainer import DtPeReportTrainer
from src.specific.dt.trainer.pe_dt_train_result import DtPeTrainResult
from src.specific.dt.trainer.pe_dt_model_trainer import DtPeModelTrainer
from src.specific.dt.trainer.pe_dt_data_trainer import DtPeDataTrainer


class TestDtPeReportTrainer(unittest.TestCase):

    def setUp(self):
        # Create real instances of the expected types without calling their __init__
        self.row_selector = DtPeDataTrainer()
        self.matrix_builder = DtPeModelTrainer()

        self.trainer = DtPeReportTrainer(self.row_selector, self.matrix_builder)

    def test_init_sets_dependencies(self):
        self.assertIs(self.row_selector, self.trainer.row_selector)
        self.assertIs(self.matrix_builder, self.trainer.matrix_builder)

    def test_init_raises_when_row_selector_is_none(self):
        with self.assertRaises((TypeError, ValueError)):
            DtPeReportTrainer(None, self.matrix_builder)

    def test_init_raises_when_matrix_builder_is_none(self):
        with self.assertRaises((TypeError, ValueError)):
            DtPeReportTrainer(self.row_selector, None)

    def test_init_raises_when_row_selector_wrong_type(self):
        with self.assertRaises((TypeError, ValueError)):
            DtPeReportTrainer("not-a-row-selector", self.matrix_builder)

    def test_init_raises_when_matrix_builder_wrong_type(self):
        with self.assertRaises((TypeError, ValueError)):
            DtPeReportTrainer(self.row_selector, "not-a-matrix-builder")

    def test_get_confusion_matrix_calls_dependencies_and_returns_built_matrix(self):
        x = MagicMock(name="x")
        y = MagicMock(name="y")
        cv = MagicMock(name="cv")
        model = MagicMock(name="model")

        train_idx_1 = [0, 1]
        test_idx_1 = [2, 3]
        train_idx_2 = [2, 3]
        test_idx_2 = [0, 1]

        cv.split.return_value = [
            (train_idx_1, test_idx_1),
            (train_idx_2, test_idx_2),
        ]

        x_train_1 = MagicMock(name="x_train_1")
        x_test_1 = MagicMock(name="x_test_1")
        y_train_1 = [0, 1]
        y_test_1 = [1, 0]

        x_train_2 = MagicMock(name="x_train_2")
        x_test_2 = MagicMock(name="x_test_2")
        y_train_2 = [1, 0]
        y_test_2 = [0, 1]

        self.row_selector.select_train_test = MagicMock(
            side_effect=[
                (x_train_1, x_test_1),
                (y_train_1, y_test_1),
                (x_train_2, x_test_2),
                (y_train_2, y_test_2),
            ]
        )

        model.predict.side_effect = [
            [1, 1],
            [0, 0],
        ]

        expected_matrix = [[2, 1], [0, 1]]
        self.matrix_builder.build = MagicMock(return_value=expected_matrix)

        actual = self.trainer.get_confusion_matrix(model, cv, x, y)

        self.assertEqual(expected_matrix, actual)

        cv.split.assert_called_once_with(x, y)

        self.assertEqual(4, self.row_selector.select_train_test.call_count)
        self.row_selector.select_train_test.assert_any_call(x, train_idx_1, test_idx_1)
        self.row_selector.select_train_test.assert_any_call(y, train_idx_1, test_idx_1)
        self.row_selector.select_train_test.assert_any_call(x, train_idx_2, test_idx_2)
        self.row_selector.select_train_test.assert_any_call(y, train_idx_2, test_idx_2)

        self.assertEqual(2, model.fit.call_count)
        model.fit.assert_any_call(x_train_1, y_train_1)
        model.fit.assert_any_call(x_train_2, y_train_2)

        self.assertEqual(2, model.predict.call_count)
        model.predict.assert_any_call(x_test_1)
        model.predict.assert_any_call(x_test_2)

        self.matrix_builder.build.assert_called_once_with(
            [1, 0, 0, 1],
            [1, 1, 0, 0]
        )

    def test_get_report_returns_dt_pe_train_result(self):
        args = MagicMock(name="args")
        ml_features = MagicMock(name="ml_features")
        model = MagicMock(name="model")
        con_matrix = MagicMock(name="con_matrix")
        scores = {
            "test_accuracy": [0.8, 0.9],
            "test_roc_auc": [0.85, 0.95],
        }

        actual = self.trainer.get_report(args, ml_features, model, con_matrix, scores)

        self.assertIsInstance(actual, DtPeTrainResult)
        self.assertIs(args, actual.input_args)
        self.assertIs(ml_features, actual.input_features)
        self.assertIs(model, actual.dt_model)
        self.assertIs(con_matrix, actual.confusion_matrix)
        self.assertEqual(scores["test_accuracy"], actual.acc_score)
        self.assertEqual(scores["test_roc_auc"], actual.auc_score)
