import unittest

from src.specific.ml.trainer.pe_ml_trainer_provider import MlPeTrainerProvider
from src.specific.ml.trainer.pe_ml_data_trainer import MlPeDataTrainer
from src.specific.ml.trainer.pe_ml_model_trainer import MlPeModelTrainer
from src.specific.ml.trainer.pe_ml_report_trainer import MlPeReportTrainer
from src.specific.ml.trainer.pe_ml_logic_trainer import MlPeLogicTrainer
from src.specific.ml.trainer.pe_ml_train_output_mapper import MlPeTrainOutputMapper
from src.specific.ml.trainer.pe_ml_train_output_writer import MlPeTrainOutputWriter
from src.specific.ml.trainer.pe_ml_train_writer import MlPeTrainWriter


class TestMlPeTrainerProvider(unittest.TestCase):

    def test_get_data_trainer_returns_ml_pe_data_trainer(self):
        trainer = MlPeTrainerProvider.get_data_trainer()

        self.assertIsInstance(trainer, MlPeDataTrainer)

    def test_get_model_trainer_returns_ml_pe_model_trainer(self):
        trainer = MlPeTrainerProvider.get_model_trainer()

        self.assertIsInstance(trainer, MlPeModelTrainer)

    def test_get_report_trainer_returns_ml_pe_report_trainer(self):
        trainer = MlPeTrainerProvider.get_report_trainer()

        self.assertIsInstance(trainer, MlPeReportTrainer)

    def test_get_logic_trainer_returns_ml_pe_logic_trainer(self):
        logic_trainer = MlPeTrainerProvider.get_logic_trainer()

        self.assertIsInstance(logic_trainer, MlPeLogicTrainer)

    def test_get_logic_trainer_has_data_trainer(self):
        logic_trainer = MlPeTrainerProvider.get_logic_trainer()

        self.assertIsInstance(logic_trainer.data_reader, MlPeDataTrainer)

    def test_get_logic_trainer_has_model_trainer(self):
        logic_trainer = MlPeTrainerProvider.get_logic_trainer()

        self.assertIsInstance(logic_trainer.trainer, MlPeModelTrainer)

    def test_get_logic_trainer_has_report_trainer(self):
        logic_trainer = MlPeTrainerProvider.get_logic_trainer()

        self.assertIsInstance(logic_trainer.reporter, MlPeReportTrainer)

    def test_get_output_mapper_returns_ml_pe_train_output_mapper(self):
        mapper = MlPeTrainerProvider.get_output_mapper()

        self.assertIsInstance(mapper, MlPeTrainOutputMapper)

    def test_get_output_writer_returns_ml_pe_train_output_writer(self):
        writer = MlPeTrainerProvider.get_output_writer()

        self.assertIsInstance(writer, MlPeTrainOutputWriter)

    def test_get_train_writer_returns_ml_pe_train_writer(self):
        writer = MlPeTrainerProvider.get_train_writer()

        self.assertIsInstance(writer, MlPeTrainWriter)

    def test_get_train_writer_has_mapper(self):
        writer = MlPeTrainerProvider.get_train_writer()

        self.assertIsInstance(writer.output_mapper, MlPeTrainOutputMapper)

    def test_get_train_writer_has_output_writer(self):
        writer = MlPeTrainerProvider.get_train_writer()

        self.assertIsInstance(writer.output_writer, MlPeTrainOutputWriter)
