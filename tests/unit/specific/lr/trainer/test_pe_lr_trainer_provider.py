import unittest

from src.specific.lr.trainer.pe_lr_trainer_provider import LrPeTrainerProvider
from src.specific.lr.trainer.pe_lr_data_trainer import LrPeDataTrainer
from src.specific.lr.trainer.pe_lr_model_trainer import LrPeModelTrainer
from src.specific.lr.trainer.pe_lr_report_trainer import LrPeReportTrainer
from src.specific.lr.trainer.pe_lr_logic_trainer import LrPeLogicTrainer
from src.specific.lr.trainer.pe_lr_train_output_mapper import LrPeTrainOutputMapper
from src.specific.lr.trainer.pe_lr_train_output_writer import LrPeTrainOutputWriter
from src.specific.lr.trainer.pe_lr_train_writer import LrPeTrainWriter


class TestLrPeTrainerProvider(unittest.TestCase):

    def test_get_data_trainer_returns_lr_pe_data_trainer(self):
        actual = LrPeTrainerProvider.get_data_trainer()

        self.assertIsNotNone(actual)
        self.assertIsInstance(actual, LrPeDataTrainer)

    def test_get_model_trainer_returns_lr_pe_model_trainer(self):
        actual = LrPeTrainerProvider.get_model_trainer()

        self.assertIsNotNone(actual)
        self.assertIsInstance(actual, LrPeModelTrainer)

    def test_get_report_trainer_returns_lr_pe_report_trainer(self):
        actual = LrPeTrainerProvider.get_report_trainer()

        self.assertIsNotNone(actual)
        self.assertIsInstance(actual, LrPeReportTrainer)

    def test_get_logic_trainer_returns_lr_pe_logic_trainer(self):
        actual = LrPeTrainerProvider.get_logic_trainer()

        self.assertIsNotNone(actual)
        self.assertIsInstance(actual, LrPeLogicTrainer)

    def test_get_logic_trainer_creates_new_trainers(self):
        first = LrPeTrainerProvider.get_logic_trainer()
        second = LrPeTrainerProvider.get_logic_trainer()

        self.assertIsNot(first, second)
        self.assertIsNot(first.data_reader, second.data_reader)
        self.assertIsNot(first.trainer, second.trainer)
        self.assertIsNot(first.reporter, second.reporter)

    def test_get_output_mapper_returns_lr_pe_train_output_mapper(self):
        actual = LrPeTrainerProvider.get_output_mapper()

        self.assertIsNotNone(actual)
        self.assertIsInstance(actual, LrPeTrainOutputMapper)

    def test_get_output_writer_returns_lr_pe_train_output_writer(self):
        actual = LrPeTrainerProvider.get_output_writer()

        self.assertIsNotNone(actual)
        self.assertIsInstance(actual, LrPeTrainOutputWriter)

    def test_get_train_writer_returns_lr_pe_train_writer(self):
        actual = LrPeTrainerProvider.get_train_writer()

        self.assertIsNotNone(actual)
        self.assertIsInstance(actual, LrPeTrainWriter)

    def test_get_train_writer_creates_new_instances(self):
        first = LrPeTrainerProvider.get_train_writer()
        second = LrPeTrainerProvider.get_train_writer()

        self.assertIsNot(first, second)
        self.assertIsNot(first.output_mapper, second.output_mapper)
        self.assertIsNot(first.output_writer, second.output_writer)

    def test_get_logic_trainer_builds_expected_dependency_types(self):
        actual = LrPeTrainerProvider.get_logic_trainer()

        self.assertIsNotNone(actual)
        self.assertIsInstance(actual, LrPeLogicTrainer)
        self.assertIsInstance(actual.data_reader, LrPeDataTrainer)
        self.assertIsInstance(actual.trainer, LrPeModelTrainer)
        self.assertIsInstance(actual.reporter, LrPeReportTrainer)
