import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from src.specific.dt.evaluate.pe_dt_evaluator_reader import DtPeEvaluatorReader
from src.specific.dt.evaluate.file_util import FileUtil


class TestDtPeEvaluatorReader(unittest.TestCase):

    def setUp(self):
        self.validator = MagicMock(spec=FileUtil)
        self.reader = DtPeEvaluatorReader(self.validator)

    def test_validate_output_calls_validate_directory_with_expected_args(self):
        args_out = MagicMock()
        args_out.output_dir = "output_dir"

        self.reader.validate_output(args_out)

        self.validator.validate_directory.assert_called_once_with("output_dir", True, True)

    def test_validate_output_raises_when_validator_raises(self):
        args_out = MagicMock()
        args_out.output_dir = "output_dir"
        self.validator.validate_directory.side_effect = FileNotFoundError("Not writeable file: output_dir")

        with self.assertRaises(FileNotFoundError) as context:
            self.reader.validate_output(args_out)

        self.validator.validate_directory.assert_called_once_with("output_dir", True, True)
        self.assertEqual("Not writeable file: output_dir", str(context.exception))

    def test_validate_input_calls_validate_directory_with_expected_args(self):
        args_in = MagicMock()
        args_in.input_dir = "input_dir"

        self.reader.validate_input(args_in)

        self.validator.validate_directory.assert_called_once_with("input_dir", True, False)

    def test_validate_input_raises_when_validator_raises(self):
        args_in = MagicMock()
        args_in.input_dir = "input_dir"
        self.validator.validate_directory.side_effect = FileNotFoundError("Not found file: input_dir")

        with self.assertRaises(FileNotFoundError) as context:
            self.reader.validate_input(args_in)

        self.validator.validate_directory.assert_called_once_with("input_dir", True, False)
        self.assertEqual("Not found file: input_dir", str(context.exception))

    def test_read_dt_model_from_joblib_file_calls_load_with_expected_path(self):
        args_in = MagicMock()
        args_in.input_model = "model.joblib"

        with patch("src.specific.dt.evaluate.pe_dt_evaluator_reader.load") as mock_load:
            mock_model = MagicMock(name="model")
            mock_load.return_value = mock_model

            result = self.reader.read_dt_model_from_joblib_file(args_in)

            mock_load.assert_called_once_with("model.joblib")
            self.assertIs(result, mock_model)

    def test_read_csv_to_df_calls_read_csv_with_expected_path(self):
        input_csv = "data.csv"

        with patch("src.specific.dt.evaluate.pe_dt_evaluator_reader.pd.read_csv") as mock_read_csv:
            mock_df = pd.DataFrame({"a": [1, 2]})
            mock_read_csv.return_value = mock_df

            result = self.reader.read_csv_to_df(input_csv)

            mock_read_csv.assert_called_once_with("data.csv")
            self.assertIs(result, mock_df)
