import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from src.common.validator.file_validator import FileValidator
from src.specific.rf.evaluate.pe_rf_evaluator_reader import RfPeEvaluatorReader


class TestRfPeEvaluatorReader(unittest.TestCase):

    def setUp(self):
        self.validator = MagicMock(spec=FileValidator)
        self.reader = RfPeEvaluatorReader(self.validator)

    @patch('src.specific.rf.evaluate.pe_rf_evaluator_reader.load')
    def test_read_rf_model_from_joblib_file_loads_model(self, mock_load):
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        args_input = SimpleNamespace(input_model="path/to/model.joblib")
        actual = self.reader.read_rf_model_from_joblib_file(args_input)

        self.assertIs(actual, mock_model)
        mock_load.assert_called_once_with("path/to/model.joblib")

    def test_read_csv_to_df_loads_data(self):
        mock_df = pd.DataFrame({
            "f1": [1, 2, 3],
            "Label": [0, 1, 0]
        })
        
        with patch('src.specific.rf.evaluate.pe_rf_evaluator_reader.pd.read_csv', return_value=mock_df):
            actual = self.reader.read_csv_to_df("path/to/test.csv")

        self.assertIsInstance(actual, pd.DataFrame)
        self.assertEqual(len(actual), 3)
