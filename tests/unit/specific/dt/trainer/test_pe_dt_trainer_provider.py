import unittest

from src.specific.dt.trainer.pe_dt_trainer_provider import DtPeTrainerProvider
from src.specific.dt.trainer.pe_dt_io_trainer import DtPeIoTrainer
from src.specific.dt.trainer.pe_dt_data_trainer import DtPeDataTrainer
from src.specific.dt.trainer.pe_dt_model_trainer import DtPeModelTrainer
from src.specific.dt.trainer.pe_dt_report_trainer import DtPeReportTrainer
from src.specific.dt.trainer.pe_dt_logic_trainer import DtPeLogicTrainer
from src.specific.dt.trainer.pe_dt_train_output_mapper import DtPeTrainOutputMapper
from src.specific.dt.trainer.pe_dt_train_output_writer import DtPeTrainOutputWriter
from src.specific.dt.trainer.pe_dt_train_writer import DtPeTrainWriter


class TestDtPeTrainerProvider(unittest.TestCase):

    def test_get_io_trainer_returns_dt_pe_io_trainer(self):
        actual = DtPeTrainerProvider.get_io_trainer()

        self.assertIsNotNone(actual)
        self.assertIsInstance(actual, DtPeIoTrainer)

    def test_get_io_trainer_returns_new_object_graph_each_time(self):
        first = DtPeTrainerProvider.get_io_trainer()
        second = DtPeTrainerProvider.get_io_trainer()

        self.assertIsNot(first, second)
        self.assertIsNot(first.reader, second.reader)
        self.assertIsNot(first.trainer, second.trainer)
        self.assertIsNot(first.writer, second.writer)

    def test_get_io_trainer_builds_expected_direct_dependency_types(self):
        actual = DtPeTrainerProvider.get_io_trainer()

        self.assertIsNotNone(actual)
        self.assertIsNotNone(actual.reader)
        self.assertIsNotNone(actual.trainer)
        self.assertIsNotNone(actual.writer)

        self.assertIsInstance(actual.reader, DtPeDataTrainer)
        self.assertIsInstance(actual.trainer, DtPeLogicTrainer)
        self.assertIsInstance(actual.writer, DtPeTrainWriter)

    def test_get_io_trainer_builds_expected_trainer_dependency_types(self):
        actual = DtPeTrainerProvider.get_io_trainer()

        self.assertIsNotNone(actual)
        self.assertIsNotNone(actual.trainer)
        self.assertIsInstance(actual.trainer, DtPeLogicTrainer)

        # actual.trainer

        self.assertIsNotNone(actual.trainer.data_reader)
        self.assertIsNotNone(actual.trainer.trainer)
        self.assertIsNotNone(actual.trainer.reporter)

        self.assertIsInstance(actual.trainer.data_reader, DtPeDataTrainer)
        self.assertIsInstance(actual.trainer.trainer, DtPeModelTrainer)
        self.assertIsInstance(actual.trainer.reporter, DtPeReportTrainer)

        self.assertIsInstance(actual.trainer.reporter.row_selector, DtPeDataTrainer)
        self.assertIsInstance(actual.trainer.reporter.matrix_builder, DtPeModelTrainer)

    def test_get_io_trainer_builds_expected_writer_dependency_types(self):
        actual = DtPeTrainerProvider.get_io_trainer()

        self.assertIsNotNone(actual)
        self.assertIsNotNone(actual.writer)
        self.assertIsInstance(actual.writer, DtPeTrainWriter)

        self.assertIsInstance(actual.writer.mapper, DtPeTrainOutputMapper)
        self.assertIsInstance(actual.writer.writer, DtPeTrainOutputWriter)
