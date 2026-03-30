import unittest

from src.specific.rf.trainer.pe_rf_trainer_provider import RfPeTrainerProvider
from src.specific.rf.trainer.pe_rf_io_trainer import RfPeIoTrainer
from src.specific.rf.trainer.pe_rf_data_trainer import RfPeDataTrainer
from src.specific.rf.trainer.pe_rf_model_trainer import RfPeModelTrainer
from src.specific.rf.trainer.pe_rf_report_trainer import RfPeReportTrainer
from src.specific.rf.trainer.pe_rf_logic_trainer import RfPeLogicTrainer
from src.specific.rf.trainer.pe_rf_train_output_mapper import RfPeTrainOutputMapper
from src.specific.rf.trainer.pe_rf_train_output_writer import RfPeTrainOutputWriter
from src.specific.rf.trainer.pe_rf_train_writer import RfPeTrainWriter


class TestRfPeTrainerProvider(unittest.TestCase):

    def test_get_io_trainer_returns_rf_pe_io_trainer(self):
        actual = RfPeTrainerProvider.get_io_trainer()

        self.assertIsNotNone(actual)
        self.assertIsInstance(actual, RfPeIoTrainer)

    def test_get_io_trainer_returns_new_object_graph_each_time(self):
        first = RfPeTrainerProvider.get_io_trainer()
        second = RfPeTrainerProvider.get_io_trainer()

        self.assertIsNot(first, second)
        self.assertIsNot(first.reader, second.reader)
        self.assertIsNot(first.trainer, second.trainer)
        self.assertIsNot(first.writer, second.writer)

    def test_get_io_trainer_builds_expected_direct_dependency_types(self):
        actual = RfPeTrainerProvider.get_io_trainer()

        self.assertIsNotNone(actual)
        self.assertIsNotNone(actual.reader)
        self.assertIsNotNone(actual.trainer)
        self.assertIsNotNone(actual.writer)

        self.assertIsInstance(actual.reader, RfPeDataTrainer)
        self.assertIsInstance(actual.trainer, RfPeLogicTrainer)
        self.assertIsInstance(actual.writer, RfPeTrainWriter)

    def test_get_io_trainer_builds_expected_trainer_dependency_types(self):
        actual = RfPeTrainerProvider.get_io_trainer()

        self.assertIsNotNone(actual)
        self.assertIsNotNone(actual.trainer)
        self.assertIsInstance(actual.trainer, RfPeLogicTrainer)

        self.assertIsNotNone(actual.trainer.data_reader)
        self.assertIsNotNone(actual.trainer.trainer)
        self.assertIsNotNone(actual.trainer.reporter)

        self.assertIsInstance(actual.trainer.data_reader, RfPeDataTrainer)
        self.assertIsInstance(actual.trainer.trainer, RfPeModelTrainer)
        self.assertIsInstance(actual.trainer.reporter, RfPeReportTrainer)

        self.assertIsInstance(actual.trainer.reporter.row_selector, RfPeDataTrainer)
        self.assertIsInstance(actual.trainer.reporter.matrix_builder, RfPeModelTrainer)

    def test_get_io_trainer_builds_expected_writer_dependency_types(self):
        actual = RfPeTrainerProvider.get_io_trainer()

        self.assertIsNotNone(actual)
        self.assertIsNotNone(actual.writer)
        self.assertIsInstance(actual.writer, RfPeTrainWriter)

        self.assertIsInstance(actual.writer.mapper, RfPeTrainOutputMapper)
        self.assertIsInstance(actual.writer.writer, RfPeTrainOutputWriter)
