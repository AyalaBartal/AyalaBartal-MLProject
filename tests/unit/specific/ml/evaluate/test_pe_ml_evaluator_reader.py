import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pandas as pd

from src.specific.ml.evaluate.pe_ml_evaluator_reader import MlPeEvaluatorReader
from src.common.validator.file_validator import FileValidator


class TestMlPeEvaluatorReader(unittest.TestCase):

    def setUp(self):
        self.validator = MagicMock(spec=FileValidator)
        self.reader = MlPeEvaluatorReader(validator=self.validator)

    def test_init_stores_validator(self):
        self.assertIs(self.validator, self.reader.validator)

    def test_validate_output_calls_validator(self):
        args_out = SimpleNamespace(output_dir="/output")

        self.reader.validate_output(args_out)

        self.validator.validate_directory.assert_called_once_with("/output", True, True)

    def test_validate_input_calls_validator(self):
        args_in = SimpleNamespace(input_dir="/input")

        self.reader.validate_input(args_in)

        self.validator.validate_directory.assert_called_once_with("/input", True, False)

    @patch('torch.load')
    def test_read_ml_model_from_pt_file_loads_model(self, mock_load):
        import torch
        args_in = SimpleNamespace(input_model="model.pt")
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        result = self.reader.read_ml_model_from_pt_file(args_in)

        self.assertIs(result, mock_model)

    def test_read_csv_to_df_returns_dataframe(self):
        df = pd.DataFrame({"f1": [1, 2], "label": [0, 1]})
        
        with patch('pandas.read_csv', return_value=df):
            result = self.reader.read_csv_to_df("test.csv")

            self.assertIsInstance(result, pd.DataFrame)

    def test_read_csv_to_df_returns_correct_data(self):
        df = pd.DataFrame({
            "feature1": [10, 20, 30],
            "feature2": [40, 50, 60],
            "label": [0, 1, 0]
        })

        with patch('pandas.read_csv', return_value=df):
            result = self.reader.read_csv_to_df("data.csv")

            self.assertEqual(len(result), 3)
            self.assertEqual(len(result.columns), 3)
