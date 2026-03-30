import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.specific.rf.trainer.pe_rf_train_output_mapper import RfPeTrainOutputMapper
from src.specific.rf.trainer.pe_rf_train_result import RfPeTrainResult


class TestRfPeTrainOutputMapper(unittest.TestCase):

    def setUp(self):
        self.mapper = RfPeTrainOutputMapper()

    def test_get_report_returns_dict_with_metrics(self):
        args = SimpleNamespace(n_splits=5, n_estimators=100, max_depth=10)
        features = pd.DataFrame({
            "f1": [1, 2, 3],
            "f2": [4, 5, 6]
        })
        model = None
        cm = np.array([[100, 10], [15, 200]])
        auc_score = np.array([0.85, 0.87, 0.86])
        acc_score = np.array([0.80, 0.82, 0.81])
        
        result = RfPeTrainResult(args, features, model, cm, acc_score, auc_score)
        
        actual = self.mapper.get_report(args, result)

        self.assertIsInstance(actual, dict)
        self.assertIn("auc_mean", actual)
        self.assertIn("auc_std", actual)
        self.assertIn("acc_mean", actual)
        self.assertIn("acc_std", actual)
        self.assertIn("n_features", actual)
        self.assertIn("n_samples", actual)

    def test_get_feature_columns_list_returns_list(self):
        args = SimpleNamespace()
        features = pd.DataFrame({
            "feature_1": [1, 2, 3],
            "feature_2": [4, 5, 6]
        })
        result = RfPeTrainResult(args, features, None, None, None, None)

        actual = self.mapper.get_feature_columns_list(result)

        self.assertIsInstance(actual, list)
        self.assertEqual(len(actual), 2)
        self.assertIn("feature_1", actual)
        self.assertIn("feature_2", actual)

    def test_get_end_message_returns_string(self):
        args = SimpleNamespace(n_splits=5, n_estimators=100, max_depth=10)
        features = pd.DataFrame({
            "f1": [1, 2, 3],
            "f2": [4, 5, 6]
        })
        cm = np.array([[100, 10], [15, 200]])
        auc_score = np.array([0.85, 0.87, 0.86])
        acc_score = np.array([0.80, 0.82, 0.81])
        
        result = RfPeTrainResult(args, features, None, cm, acc_score, auc_score)
        
        actual = self.mapper.get_end_message(result)

        self.assertIsInstance(actual, str)
        self.assertIn("Random Forest", actual)
        self.assertIn("Cross-Validation", actual)
