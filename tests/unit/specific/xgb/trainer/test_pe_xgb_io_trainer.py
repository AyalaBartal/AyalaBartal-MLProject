import unittest
from unittest.mock import MagicMock
import pandas as pd

from src.specific.xgb.trainer.pe_xgb_io_trainer import XgbPeIoTrainer


class TestXgbPeIoTrainer(unittest.TestCase):

    def setUp(self):
        self.reader = MagicMock()
        self.trainer = MagicMock()
        self.writer = MagicMock()

        self.io_trainer = XgbPeIoTrainer(
            reader=self.reader,
            trainer=self.trainer,
            writer=self.writer
        )

    def test_init_sets_dependencies(self):
        self.assertIs(self.reader, self.io_trainer.reader)
        self.assertIs(self.trainer, self.io_trainer.trainer)
        self.assertIs(self.writer, self.io_trainer.writer)

    def test_train_reads_csv_trains_and_writes(self):
        algo_args = MagicMock()
        report_args = MagicMock()

        df = pd.DataFrame({
            "f1": [1, 2, 3],
            "f2": [4, 5, 6],
            "label": [0, 1, 0]
        })

        result = MagicMock()

        self.reader.read_csv_to_df.return_value = df
        self.trainer.train.return_value = result

        self.io_trainer.train(algo_args, report_args)

        self.reader.read_csv_to_df.assert_called_once()
        self.trainer.train.assert_called_once_with(algo_args, df)
        self.writer.write_output.assert_called_once_with(report_args, result)

    def test_train_calls_components_in_order(self):
        algo_args = MagicMock()
        report_args = MagicMock()

        df = pd.DataFrame({"f1": [1], "label": [0]})
        result = MagicMock()

        self.reader.read_csv_to_df.return_value = df
        self.trainer.train.return_value = result

        self.io_trainer.train(algo_args, report_args)

        # Verify each step called
        self.reader.read_csv_to_df.assert_called()
        self.trainer.train.assert_called()
        self.writer.write_output.assert_called()


if __name__ == "__main__":
    unittest.main()

