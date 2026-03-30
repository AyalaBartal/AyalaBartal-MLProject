import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from src.specific.rf.trainer.pe_rf_data_trainer import RfPeDataTrainer
from src.specific.rf.trainer.pe_rf_model_trainer import RfPeModelTrainer
from src.specific.rf.trainer.pe_rf_report_trainer import RfPeReportTrainer


class TestRfPeReportTrainer(unittest.TestCase):

    def setUp(self):
        self.mock_row_selector = MagicMock(spec=RfPeDataTrainer)
        self.mock_matrix_builder = MagicMock(spec=RfPeModelTrainer)
        self.reporter = RfPeReportTrainer(self.mock_row_selector, self.mock_matrix_builder)

    def test_get_confusion_matrix_calls_model_fit_and_predict(self):
        model = MagicMock()
        cv = MagicMock()
        cv.split.return_value = [([0, 1], [2, 3])]
        
        import pandas as pd
        x = pd.DataFrame({"f1": [1, 2, 3, 4]})
        y = pd.Series([0, 1, 1, 0])
        
        mock_x_train = pd.DataFrame({"f1": [1, 2]})
        mock_x_test = pd.DataFrame({"f1": [3, 4]})
        mock_y_train = pd.Series([0, 1])
        mock_y_test = pd.Series([1, 0])
        
        self.mock_row_selector.select_train_test.side_effect = [
            (mock_x_train, mock_x_test),
            (mock_y_train, mock_y_test)
        ]
        
        cm = np.array([[1, 0], [1, 1]])
        self.mock_matrix_builder.build.return_value = cm
        
        result = self.reporter.get_confusion_matrix(model, cv, x, y)
        
        np.testing.assert_array_equal(result, cm)
        self.mock_matrix_builder.build.assert_called_once()

    def test_get_report_returns_train_result(self):
        args = SimpleNamespace()
        ml_features = MagicMock()
        model = MagicMock()
        con_matrix = np.array([[2, 0], [0, 2]])
        scores = {
            'test_accuracy': np.array([0.8, 0.9]),
            'test_roc_auc': np.array([0.85, 0.95])
        }
        
        result = self.reporter.get_report(args, ml_features, model, con_matrix, scores)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.confusion_matrix.tolist(), [[2, 0], [0, 2]])
