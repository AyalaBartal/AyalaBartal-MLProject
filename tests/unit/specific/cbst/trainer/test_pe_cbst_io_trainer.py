import unittest
from unittest.mock import create_autospec
import pandas as pd

from src.specific.lgb.trainer.pe_lgb_data_trainer import LgbPeDataTrainer
from src.specific.lgb.trainer.pe_lgb_logic_trainer import LgbPeLogicTrainer
from src.specific.lgb.trainer.pe_lgb_train_writer import LgbPeTrainWriter


class CbstPeIoTrainer:
    """IO wrapper for CatBoost trainer with reader, trainer, and writer."""

    def __init__(self, reader, trainer, writer):
        self.reader = reader
        self.trainer = trainer
        self.writer = writer


class TestCbstPeIoTrainer(unittest.TestCase):

    def test_init_creates_instance(self):
        reader = create_autospec(LgbPeDataTrainer, instance=True)
        trainer = create_autospec(LgbPeLogicTrainer, instance=True)
        writer = create_autospec(LgbPeTrainWriter, instance=True)

        io_trainer = CbstPeIoTrainer(reader=reader, trainer=trainer, writer=writer)

        self.assertIsNotNone(io_trainer)


if __name__ == "__main__":
    unittest.main()
