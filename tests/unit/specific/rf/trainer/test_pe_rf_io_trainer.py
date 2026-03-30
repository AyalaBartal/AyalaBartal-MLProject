import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from src.common.validator.args_validator import ArgsValidator
from src.specific.rf.trainer.pe_rf_io_trainer import RfPeIoTrainer
from src.specific.rf.trainer.pe_rf_data_trainer import RfPeDataTrainer
from src.specific.rf.trainer.pe_rf_logic_trainer import RfPeLogicTrainer
from src.specific.rf.trainer.pe_rf_train_writer import RfPeTrainWriter


class TestRfPeIoTrainer(unittest.TestCase):

    def setUp(self):
        self.mock_reader = MagicMock(spec=RfPeDataTrainer)
        self.mock_trainer = MagicMock(spec=RfPeLogicTrainer)
        self.mock_writer = MagicMock(spec=RfPeTrainWriter)
        self.io_trainer = RfPeIoTrainer(self.mock_reader, self.mock_trainer, self.mock_writer)

    def test_train_calls_reader_trainer_and_writer(self):
        algo_args = SimpleNamespace()
        report_args = SimpleNamespace()
        
        mock_data = {"features": pd.DataFrame({"f1": [1, 2, 3]}), "labels": pd.Series([0, 1, 0])}
        mock_result = MagicMock()
        
        self.mock_reader.read_csv_to_df.return_value = mock_data
        self.mock_trainer.train.return_value = mock_result

        self.io_trainer.train(algo_args, report_args)

        self.mock_reader.read_csv_to_df.assert_called_once_with(report_args)
        self.mock_trainer.train.assert_called_once_with(algo_args, mock_data)
        self.mock_writer.write_output.assert_called_once_with(report_args, mock_result)
