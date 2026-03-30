import unittest
from unittest.mock import create_autospec

from src.specific.ml.trainer.pe_ml_report_trainer import MlPeReportTrainer
from src.specific.ml.trainer.pe_ml_data_trainer import MlPeDataTrainer
from src.specific.ml.trainer.pe_ml_model_trainer import MlPeModelTrainer
from src.specific.ml.trainer.pe_ml_logic_trainer import MlPeLogicTrainer


class TestMlPeLogicTrainer(unittest.TestCase):

    def setUp(self):
        self.reader = create_autospec(MlPeDataTrainer, instance=True)
        self.trainer = create_autospec(MlPeModelTrainer, instance=True)
        self.reporter = create_autospec(MlPeReportTrainer, instance=True)

        self.logic_trainer = MlPeLogicTrainer(
            reader=self.reader,
            trainer=self.trainer,
            reporter=self.reporter,
        )

    def test_init_sets_dependencies(self):
        self.assertIs(self.reader, self.logic_trainer.data_reader)
        self.assertIs(self.trainer, self.logic_trainer.trainer)
        self.assertIs(self.reporter, self.logic_trainer.reporter)
