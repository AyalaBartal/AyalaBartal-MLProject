import unittest
from unittest.mock import MagicMock, patch
import pandas as pd

from src.specific.xgb.evaluate.pe_xgb_evaluator_reader import XgbPeEvaluatorReader
from src.specific.xgb.evaluate.pe_xgb_evaluate_input_args import XgbPeEvaluateInputArgs


class TestXgbPeEvaluatorReader(unittest.TestCase):

    def setUp(self):
        self.validator = MagicMock()
        self.reader = XgbPeEvaluatorReader(self.validator)

    def test_init_sets_validator(self):
        self.assertIs(self.validator, self.reader.validator)

    def test_validate_input_calls_validator(self):
        args_input = XgbPeEvaluateInputArgs("input_dir")

        self.reader.validate_input(args_input)

        self.validator.validate_directory.assert_called_once_with(args_input.input_dir, True, False)

    def test_validate_output_calls_validator(self):
        args_output = MagicMock()
        args_output.output_dir = "/path/to/output"

        self.reader.validate_output(args_output)

        self.validator.validate_directory.assert_called_once_with("/path/to/output", True, True)

    @patch('src.specific.xgb.evaluate.pe_xgb_evaluator_reader.pd')
    def test_read_csv_to_df_returns_dataframe(self, mock_pd):
        expected_df = pd.DataFrame({"f1": [1, 2], "label": [0, 1]})
        mock_pd.read_csv.return_value = expected_df

        actual = self.reader.read_csv_to_df("path/to/file.csv")

        self.assertIs(actual, expected_df)

    @patch('src.specific.xgb.evaluate.pe_xgb_evaluator_reader.load')
    def test_read_xgb_model_from_joblib_file_loads_model(self, mock_load):
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        args_input = XgbPeEvaluateInputArgs("path/to/input")
        actual = self.reader.read_xgb_model_from_joblib_file(args_input)

        self.assertIs(actual, mock_model)
        mock_load.assert_called_once_with(args_input.input_model)

    @patch('src.specific.xgb.evaluate.pe_xgb_evaluator_reader.load')
    def test_read_xgb_model_from_joblib_file_with_different_path(self, mock_load):
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        args_input = MagicMock()
        args_input.input_model = "custom/model/path.joblib"

        actual = self.reader.read_xgb_model_from_joblib_file(args_input)

        mock_load.assert_called_once_with("custom/model/path.joblib")

    def test_validate_input_propagates_exception(self):
        args_input = XgbPeEvaluateInputArgs("input_dir")
        self.validator.validate_directory.side_effect = FileNotFoundError("dir not found")

        with self.assertRaises(FileNotFoundError):
            self.reader.validate_input(args_input)

    def test_validate_output_propagates_exception(self):
        args_output = MagicMock()
        args_output.output_dir = "/path/to/output"
        self.validator.validate_directory.side_effect = PermissionError("not writable")

        with self.assertRaises(PermissionError):
            self.reader.validate_output(args_output)


if __name__ == "__main__":
    unittest.main()

