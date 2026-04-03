import unittest
from unittest.mock import create_autospec, Mock
from types import SimpleNamespace
import pandas as pd
import numpy as np

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

    def test_get_confusion_matrix_returns_matrix(self):
        model = Mock(name="model")
        cv = Mock(name="cv")
        x = pd.DataFrame({"f1": [1, 2, 3, 4]})
        y = pd.Series([0, 1, 0, 1])

        train_idx = [0, 1]
        test_idx = [2, 3]

        def cv_split(x_param, y_param):
            yield (train_idx, test_idx)

        cv.split.return_value = cv_split(x, y)

        x_train = x.iloc[train_idx]
        x_test = x.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        self.row_selector.select_train_test.return_value = (x_train, x_test)
        self.row_selector.select_train_test.side_effect = lambda data, ti, tei: (
            data.iloc[ti] if isinstance(data, pd.DataFrame) else data.iloc[ti],
            data.iloc[tei] if isinstance(data, pd.DataFrame) else data.iloc[tei]
        )

        fitted_model = Mock(name="fitted_model")
        model.fit.return_value = fitted_model
        fitted_model.predict.return_value = np.array([0, 1])

        cm = np.array([[1, 0], [0, 1]])
        self.matrix_builder.build.return_value = cm

        result = self.reporter.get_confusion_matrix(model, cv, x, y)

        self.assertIsNotNone(result)

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

    def test_get_report_returns_expected_structure(self):
        args = SimpleNamespace(
            label="target",
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.0,
            random_state=42,
            n_splits=10
        )

        ml_features = pd.DataFrame({
            "f1": [1, 2, 3],
            "f2": [4, 5, 6]
        })

        model = Mock(name="model")
        con_matrix = np.array([[10, 2], [1, 8]])
        scores = {
            "test_accuracy": [0.9, 0.85, 0.92],
            "test_roc_auc": [0.91, 0.88, 0.93]
        }

        result = self.reporter.get_report(args, ml_features, model, con_matrix, scores)

        self.assertIsNotNone(result)
        self.assertEqual(result.input_args, args)
        self.assertEqual(list(result.input_features.columns), ["f1", "f2"])
        self.assertIs(result.xgb_model, model)
        np.testing.assert_array_equal(result.confusion_matrix, con_matrix)
        self.assertAlmostEqual(result.acc_score, 0.9)
        self.assertAlmostEqual(result.auc_score, 0.906667, places=5)

    def test_get_report_with_single_fold_scores(self):
        args = SimpleNamespace(
            label="target",
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.0,
            random_state=42,
            n_splits=10
        )

        ml_features = pd.DataFrame({"f1": [1, 2]})
        model = Mock(name="model")
        con_matrix = np.array([[1, 0], [0, 1]])
        scores = {
            "test_accuracy": [0.95],
            "test_roc_auc": [0.98]
        }

        result = self.reporter.get_report(args, ml_features, model, con_matrix, scores)

        self.assertAlmostEqual(result.acc_score, 0.95)
        self.assertAlmostEqual(result.auc_score, 0.98)

    def test_get_report_calculates_average_scores(self):
        args = SimpleNamespace(
            label="target",
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.0,
            random_state=42,
            n_splits=10
        )

        ml_features = pd.DataFrame({"f1": [1, 2, 3, 4, 5]})
        model = Mock(name="model")
        con_matrix = np.array([[2, 0], [0, 3]])
        scores = {
            "test_accuracy": [0.8, 0.9, 1.0],
            "test_roc_auc": [0.75, 0.85, 0.95]
        }

        result = self.reporter.get_report(args, ml_features, model, con_matrix, scores)

        # Average of [0.8, 0.9, 1.0] = 0.9
        self.assertAlmostEqual(result.acc_score, 0.9, places=5)
        # Average of [0.75, 0.85, 0.95] = 0.85
        self.assertAlmostEqual(result.auc_score, 0.85, places=5)


if __name__ == "__main__":
    unittest.main()
