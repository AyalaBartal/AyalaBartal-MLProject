import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import torch
import torch.nn as nn

from src.specific.ml.trainer.pe_ml_model_trainer import MlPeModelTrainer, MLPModel


class TestMlPeModelTrainer(unittest.TestCase):

    def setUp(self):
        self.trainer = MlPeModelTrainer()

    def test_get_mlp_model_returns_mlp_model(self):
        args = SimpleNamespace(
            hidden_sizes=[64, 32],
            random_state=42
        )

        model = self.trainer.get_mlp_model(10, args)

        self.assertIsInstance(model, MLPModel)

    def test_get_mlp_model_creates_correct_architecture(self):
        args = SimpleNamespace(
            hidden_sizes=[128, 64],
            random_state=42
        )

        model = self.trainer.get_mlp_model(20, args)

        self.assertIsNotNone(model.network)
        self.assertEqual(len(model.network), 5)

    def test_get_split_train_test_returns_stratified_kfold(self):
        from sklearn.model_selection import StratifiedKFold
        args = SimpleNamespace(
            n_splits=5,
            random_state=123
        )

        skf = self.trainer.get_split_train_test(args)

        self.assertIsInstance(skf, StratifiedKFold)
        self.assertEqual(skf.n_splits, 5)
        self.assertTrue(skf.shuffle)

    def test_fit_model_trains_and_returns_model(self):
        args = SimpleNamespace(
            epochs=1,
            batch_size=2,
            learning_rate=0.001
        )

        model = MLPModel(5, [16, 8])
        X_train = pd.DataFrame({f"f{i}": [1, 2, 3] for i in range(5)})
        y_train = pd.Series([0, 1, 0])

        trained_model = self.trainer.fit_model(model, X_train, y_train, args)

        self.assertIsInstance(trained_model, MLPModel)

    def test_fit_model_accepts_numpy_arrays(self):
        import numpy as np
        args = SimpleNamespace(
            epochs=1,
            batch_size=2,
            learning_rate=0.001
        )

        model = MLPModel(5, [16, 8])
        X_train = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]])
        y_train = np.array([0, 1, 0])

        trained_model = self.trainer.fit_model(model, X_train, y_train, args)

        self.assertIsInstance(trained_model, MLPModel)

    def test_predict_returns_numpy_array(self):
        model = MLPModel(5, [16, 8])
        X_test = pd.DataFrame({f"f{i}": [1, 2] for i in range(5)})

        predictions = self.trainer.predict(model, X_test)

        self.assertIsNotNone(predictions)

    def test_predict_proba_returns_probabilities(self):
        model = MLPModel(5, [16, 8])
        X_test = pd.DataFrame({f"f{i}": [1, 2] for i in range(5)})

        proba = self.trainer.predict_proba(model, X_test)

        self.assertEqual(proba.shape[1], 2)

    def test_predict_with_numpy_input(self):
        import numpy as np
        model = MLPModel(5, [16, 8])
        X_test = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])

        predictions = self.trainer.predict(model, X_test)

        self.assertEqual(len(predictions), 2)

    def test_build_returns_confusion_matrix_dict(self):
        y_true = [0, 1, 1, 0]
        y_pred = [0, 1, 0, 0]

        result = self.trainer.build(y_true, y_pred)

        self.assertIsInstance(result, dict)
        self.assertIn('true_neg', result)
        self.assertIn('false_pos', result)
        self.assertIn('false_neg', result)
        self.assertIn('true_pos', result)

    def test_mlp_model_forward_pass_returns_logits(self):
        model = MLPModel(10, [32, 16])
        X = torch.randn(4, 10)

        output = model(X)

        self.assertEqual(output.shape[0], 4)
        self.assertEqual(output.shape[1], 2)

    def test_mlp_model_with_different_layer_sizes(self):
        model = MLPModel(20, [256, 128, 64])

        self.assertIsNotNone(model.network)
        self.assertTrue(len(model.network) > 5)
