import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from src.specific.dt.trainer.pe_dt_model_trainer import DtPeModelTrainer


class TestDtPeModelTrainer(unittest.TestCase):

    def setUp(self):
        self.trainer = DtPeModelTrainer()

    def test_get_decision_tree_classifier_returns_expected_model(self):
        args = SimpleNamespace(
            criterion="gini",
            max_depth=5,
            min_samples_leaf=2,
            random_state=42
        )

        model = self.trainer.get_decision_tree_classifier(args)

        self.assertIsInstance(model, DecisionTreeClassifier)
        self.assertEqual(model.criterion, "gini")
        self.assertEqual(model.max_depth, 5)
        self.assertEqual(model.min_samples_leaf, 2)
        self.assertEqual(model.class_weight, "balanced")
        self.assertEqual(model.random_state, 42)

    def test_get_split_train_test_returns_expected_stratified_kfold(self):
        args = SimpleNamespace(
            n_splits=4,
            random_state=123
        )

        skf = self.trainer.get_split_train_test(args)

        self.assertIsInstance(skf, StratifiedKFold)
        self.assertEqual(skf.n_splits, 4)
        self.assertTrue(skf.shuffle)
        self.assertEqual(skf.random_state, 123)

    def test_get_cross_validate_score_returns_expected_keys(self):
        model = DecisionTreeClassifier()
        skf = StratifiedKFold(n_splits=2)

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

        # Assert structure instead of mocking internals
        self.assertIn("test_roc_auc", result)
        self.assertIn("test_accuracy", result)
        self.assertEqual(len(result["test_roc_auc"]), 2)
        self.assertEqual(len(result["test_accuracy"]), 2)

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
