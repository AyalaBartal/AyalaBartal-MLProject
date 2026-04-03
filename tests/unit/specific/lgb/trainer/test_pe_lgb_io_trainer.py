import unittest
from unittest.mock import create_autospec
import pandas as pd

from src.specific.lgb.trainer.pe_lgb_io_trainer import LgbPeIoTrainer
from src.specific.lgb.trainer.pe_lgb_data_trainer import LgbPeDataTrainer
from src.specific.lgb.trainer.pe_lgb_logic_trainer import LgbPeLogicTrainer
from src.specific.lgb.trainer.pe_lgb_train_writer import LgbPeTrainWriter


class TestLgbPeIoTrainer(unittest.TestCase):

    def test_init_creates_instance(self):
        reader = create_autospec(LgbPeDataTrainer, instance=True)
        trainer = create_autospec(LgbPeLogicTrainer, instance=True)
        writer = create_autospec(LgbPeTrainWriter, instance=True)

        io_trainer = LgbPeIoTrainer(reader=reader, trainer=trainer, writer=writer)

        self.assertIsNotNone(io_trainer)


if __name__ == "__main__":
    unittest.main()
