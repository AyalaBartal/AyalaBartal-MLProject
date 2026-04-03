import unittest
from unittest.mock import create_autospec
import pandas as pd

from src.specific.xgb.trainer.pe_xgb_io_trainer import XgbPeIoTrainer
from src.specific.xgb.trainer.pe_xgb_data_trainer import XgbPeDataTrainer
from src.specific.xgb.trainer.pe_xgb_logic_trainer import XgbPeLogicTrainer
from src.specific.xgb.trainer.pe_xgb_train_writer import XgbPeTrainWriter


class TestXgbPeIoTrainer(unittest.TestCase):

    def test_init_creates_instance(self):
        reader = create_autospec(XgbPeDataTrainer, instance=True)
        trainer = create_autospec(XgbPeLogicTrainer, instance=True)
        writer = create_autospec(XgbPeTrainWriter, instance=True)

        io_trainer = XgbPeIoTrainer(reader=reader, trainer=trainer, writer=writer)

        self.assertIsNotNone(io_trainer)


if __name__ == "__main__":
    unittest.main()


