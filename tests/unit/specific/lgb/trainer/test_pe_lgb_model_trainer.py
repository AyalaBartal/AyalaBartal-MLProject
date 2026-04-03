import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

from src.specific.lgb.trainer.pe_lgb_model_trainer import LgbPeModelTrainer


class TestLgbPeModelTrainer(unittest.TestCase):

    def setUp(self):
        self.trainer = LgbPeModelTrainer()

    def test_get_lightgbm_classifier_returns_expected_model(self):
        args = SimpleNamespace(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42
        )

        model = self.trainer.get_lightgbm_classifier(args)

        self.assertIsInstance(model, LGBMClassifier)
        self.assertEqual(model.n_estimators, 100)
        self.assertEqual(model.max_depth, 6)
        self.assertEqual(model.learning_rate, 0.1)
        self.assertEqual(model.num_leaves, 31)
        self.assertEqual(model.random_state, 42)

    def test_get_lightgbm_classifier_with_different_hyperparameters(self):
        args = SimpleNamespace(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=63,
            random_state=123
        )

        model = self.trainer.get_lightgbm_classifier(args)

        self.assertEqual(model.n_estimators, 200)
        self.assertEqual(model.max_depth, 8)
        self.assertEqual(model.learning_rate, 0.05)

    def test_get_lightgbm_classifier_instantiates_correctly(self):
        args = SimpleNamespace(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42
        )

        model = self.trainer.get_lightgbm_classifier(args)

        self.assertIsInstance(model, LGBMClassifier)

    def test_get_split_train_test_returns_expected_stratified_kfold(self):
        args = SimpleNamespace(
            n_splits=5,
            random_state=123
        )

        skf = self.trainer.get_split_train_test(args)

        self.assertIsInstance(skf, StratifiedKFold)
        self.assertEqual(skf.n_splits, 5)
        self.assertTrue(skf.shuffle)
        self.assertEqual(skf.random_state, 123)

    def test_get_split_train_test_with_different_splits(self):
        args = SimpleNamespace(
            n_splits=10,
            random_state=42
        )

        skf = self.trainer.get_split_train_test(args)

        self.assertEqual(skf.n_splits, 10)

    def test_get_split_train_test_has_shuffle_enabled(self):
        args = SimpleNamespace(
            n_splits=3,
            random_state=42
        )

        skf = self.trainer.get_split_train_test(args)

        self.assertTrue(skf.shuffle)

    def test_get_cross_validate_score_returns_expected_keys(self):
        model = LGBMClassifier(n_estimators=10, random_state=42)
        skf = StratifiedKFold(n_splits=2, random_state=42, shuffle=True)

        ml_features = pd.DataFrame({
            "f1": [1, 2, 3, 4],
            "f2": [4, 5, 6, 7]
        })
        ml_label = pd.Series([0, 1, 0, 1])

        result = self.trainer.get_cross_validate_score(
            model,
            skf,
            ml_features,
            ml_label
        )

        self.assertIn("test_roc_auc", result)
        self.assertIn("test_accuracy", result)
        self.assertEqual(len(result["test_roc_auc"]), 2)
        self.assertEqual(len(result["test_accuracy"]), 2)

    def test_get_cross_validate_score_returns_numeric_scores(self):
        model = LGBMClassifier(n_estimators=5, random_state=42)
        skf = StratifiedKFold(n_splits=2, random_state=42, shuffle=True)

        ml_features = pd.DataFrame({
            "f1": [1, 2, 3, 4, 5, 6],
            "f2": [4, 5, 6, 7, 8, 9]
        })
        ml_label = pd.Series([0, 1, 0, 1, 0, 1])

        result = self.trainer.get_cross_validate_score(
            model,
            skf,
            ml_features,
            ml_label
        )

        for score in result["test_roc_auc"]:
            self.assertIsInstance(score, (int, float, np.number))
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

        for score in result["test_accuracy"]:
            self.assertIsInstance(score, (int, float, np.number))
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_get_cross_validate_score_with_more_splits(self):
        model = LGBMClassifier(n_estimators=10, random_state=42)
        skf = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)

        ml_features = pd.DataFrame({
            "f1": list(range(30)),
            "f2": list(range(30, 60))
        })
        ml_label = pd.Series([i % 2 for i in range(30)])

        result = self.trainer.get_cross_validate_score(
            model,
            skf,
            ml_features,
            ml_label
        )

        self.assertEqual(len(result["test_roc_auc"]), 3)
        self.assertEqual(len(result["test_accuracy"]), 3)

    def test_fit_model_calls_model_fit_and_returns_fitted_model(self):
        model = MagicMock()
        ml_features = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
        ml_label = pd.Series([0, 1, 0])

        fitted_model = MagicMock()
        model.fit.return_value = fitted_model

        actual = self.trainer.fit_model(model, ml_features, ml_label)

        self.assertIs(actual, fitted_model)
        model.fit.assert_called_once_with(ml_features, ml_label)

    def test_build_returns_confusion_matrix(self):
        y_true = [0, 1, 1, 0]
        y_pred = [0, 1, 0, 0]

        actual = self.trainer.build(y_true, y_pred)

        self.assertEqual(actual.tolist(), [[2, 0], [1, 1]])

    def test_build_confusion_matrix_all_correct(self):
        y_true = [0, 1, 0, 1, 0]
        y_pred = [0, 1, 0, 1, 0]

        actual = self.trainer.build(y_true, y_pred)

        expected = [[3, 0], [0, 2]]
        self.assertEqual(actual.tolist(), expected)

    def test_build_confusion_matrix_all_wrong(self):
        y_true = [0, 0, 0, 1, 1]
        y_pred = [1, 1, 1, 0, 0]

        actual = self.trainer.build(y_true, y_pred)

        expected = [[0, 3], [2, 0]]
        self.assertEqual(actual.tolist(), expected)


if __name__ == "__main__":
    unittest.main()
