import unittest

from src.specific.xgb.trainer.pe_xgb_trainer_provider import XgbPeTrainerProvider
from src.specific.xgb.trainer.pe_xgb_io_trainer import XgbPeIoTrainer
from src.specific.xgb.trainer.pe_xgb_data_trainer import XgbPeDataTrainer
from src.specific.xgb.trainer.pe_xgb_model_trainer import XgbPeModelTrainer
from src.specific.xgb.trainer.pe_xgb_report_trainer import XgbPeReportTrainer
from src.specific.xgb.trainer.pe_xgb_logic_trainer import XgbPeLogicTrainer
from src.specific.xgb.trainer.pe_xgb_output_mapper import XgbPeOutputMapper
from src.specific.xgb.trainer.pe_xgb_output_writer import XgbPeOutputWriter
from src.specific.xgb.trainer.pe_xgb_train_writer import XgbPeTrainWriter


class TestXgbPeTrainerProvider(unittest.TestCase):

    def test_get_io_trainer_returns_xgb_pe_io_trainer(self):
        actual = XgbPeTrainerProvider.get_io_trainer()

        self.assertIsNotNone(actual)
        self.assertIsInstance(actual, XgbPeIoTrainer)

    def test_get_io_trainer_returns_new_object_graph_each_time(self):
        first = XgbPeTrainerProvider.get_io_trainer()
        second = XgbPeTrainerProvider.get_io_trainer()

        self.assertIsNot(first, second)
        self.assertIsNot(first.reader, second.reader)
        self.assertIsNot(first.trainer, second.trainer)
        self.assertIsNot(first.writer, second.writer)

    def test_get_io_trainer_builds_expected_direct_dependency_types(self):
        actual = XgbPeTrainerProvider.get_io_trainer()

        self.assertIsNotNone(actual)
        self.assertIsNotNone(actual.reader)
        self.assertIsNotNone(actual.trainer)
        self.assertIsNotNone(actual.writer)

        self.assertIsInstance(actual.reader, XgbPeDataTrainer)
        self.assertIsInstance(actual.trainer, XgbPeLogicTrainer)
        self.assertIsInstance(actual.writer, XgbPeTrainWriter)

    def test_get_io_trainer_builds_expected_trainer_dependency_types(self):
        actual = XgbPeTrainerProvider.get_io_trainer()

        self.assertIsNotNone(actual)
        self.assertIsNotNone(actual.trainer)
        self.assertIsInstance(actual.trainer, XgbPeLogicTrainer)

        self.assertIsNotNone(actual.trainer.data_reader)
        self.assertIsNotNone(actual.trainer.trainer)
        self.assertIsNotNone(actual.trainer.reporter)

        self.assertIsInstance(actual.trainer.data_reader, XgbPeDataTrainer)
        self.assertIsInstance(actual.trainer.trainer, XgbPeModelTrainer)
        self.assertIsInstance(actual.trainer.reporter, XgbPeReportTrainer)

        self.assertIsInstance(actual.trainer.reporter.row_selector, XgbPeDataTrainer)
        self.assertIsInstance(actual.trainer.reporter.matrix_builder, XgbPeModelTrainer)

    def test_get_io_trainer_builds_expected_writer_dependency_types(self):
        actual = XgbPeTrainerProvider.get_io_trainer()

        self.assertIsNotNone(actual)
        self.assertIsNotNone(actual.writer)
        self.assertIsInstance(actual.writer, XgbPeTrainWriter)

        self.assertIsInstance(actual.writer.mapper, XgbPeOutputMapper)
        self.assertIsInstance(actual.writer.writer, XgbPeOutputWriter)

    def test_get_io_trainer_graph_is_fully_connected(self):
        actual = XgbPeTrainerProvider.get_io_trainer()

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
