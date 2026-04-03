import unittest

from src.specific.lgb.trainer.pe_lgb_trainer_provider import LgbPeTrainerProvider
from src.specific.lgb.trainer.pe_lgb_io_trainer import LgbPeIoTrainer
from src.specific.lgb.trainer.pe_lgb_data_trainer import LgbPeDataTrainer
from src.specific.lgb.trainer.pe_lgb_model_trainer import LgbPeModelTrainer
from src.specific.lgb.trainer.pe_lgb_report_trainer import LgbPeReportTrainer
from src.specific.lgb.trainer.pe_lgb_logic_trainer import LgbPeLogicTrainer
from src.specific.lgb.trainer.pe_lgb_output_mapper import LgbPeOutputMapper
from src.specific.lgb.trainer.pe_lgb_output_writer import LgbPeOutputWriter
from src.specific.lgb.trainer.pe_lgb_train_writer import LgbPeTrainWriter


class TestLgbPeTrainerProvider(unittest.TestCase):

    def test_get_io_trainer_returns_lgb_pe_io_trainer(self):
        actual = LgbPeTrainerProvider.get_io_trainer()

        self.assertIsNotNone(actual)
        self.assertIsInstance(actual, LgbPeIoTrainer)

    def test_get_io_trainer_returns_new_object_graph_each_time(self):
        first = LgbPeTrainerProvider.get_io_trainer()
        second = LgbPeTrainerProvider.get_io_trainer()

        self.assertIsNot(first, second)
        self.assertIsNot(first.reader, second.reader)
        self.assertIsNot(first.trainer, second.trainer)
        self.assertIsNot(first.writer, second.writer)

    def test_get_io_trainer_builds_expected_direct_dependency_types(self):
        actual = LgbPeTrainerProvider.get_io_trainer()

        self.assertIsNotNone(actual)
        self.assertIsNotNone(actual.reader)
        self.assertIsNotNone(actual.trainer)
        self.assertIsNotNone(actual.writer)

        self.assertIsInstance(actual.reader, LgbPeDataTrainer)
        self.assertIsInstance(actual.trainer, LgbPeLogicTrainer)
        self.assertIsInstance(actual.writer, LgbPeTrainWriter)

    def test_get_io_trainer_builds_expected_trainer_dependency_types(self):
        actual = LgbPeTrainerProvider.get_io_trainer()

        self.assertIsNotNone(actual)
        self.assertIsNotNone(actual.trainer)
        self.assertIsInstance(actual.trainer, LgbPeLogicTrainer)

        self.assertIsNotNone(actual.trainer.data_reader)
        self.assertIsNotNone(actual.trainer.trainer)
        self.assertIsNotNone(actual.trainer.reporter)

        self.assertIsInstance(actual.trainer.data_reader, LgbPeDataTrainer)
        self.assertIsInstance(actual.trainer.trainer, LgbPeModelTrainer)
        self.assertIsInstance(actual.trainer.reporter, LgbPeReportTrainer)

        self.assertIsInstance(actual.trainer.reporter.row_selector, LgbPeDataTrainer)
        self.assertIsInstance(actual.trainer.reporter.matrix_builder, LgbPeModelTrainer)

    def test_get_io_trainer_builds_expected_writer_dependency_types(self):
        actual = LgbPeTrainerProvider.get_io_trainer()

        self.assertIsNotNone(actual)
        self.assertIsNotNone(actual.writer)
        self.assertIsInstance(actual.writer, LgbPeTrainWriter)

        self.assertIsInstance(actual.writer.mapper, LgbPeOutputMapper)
        self.assertIsInstance(actual.writer.writer, LgbPeOutputWriter)

    def test_get_io_trainer_graph_is_fully_connected(self):
        actual = LgbPeTrainerProvider.get_io_trainer()

        # Verify all nested dependencies are properly wired
        self.assertIsNotNone(actual.reader)
        self.assertIsNotNone(actual.trainer)
        self.assertIsNotNone(actual.trainer.data_reader)
        self.assertIsNotNone(actual.trainer.trainer)
        self.assertIsNotNone(actual.trainer.reporter)
        self.assertIsNotNone(actual.writer)
        self.assertIsNotNone(actual.writer.mapper)
        self.assertIsNotNone(actual.writer.writer)


if __name__ == "__main__":
    unittest.main()
