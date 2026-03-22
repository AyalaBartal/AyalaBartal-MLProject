import unittest
from unittest.mock import create_autospec

from src.specific.dt.trainer import DtPeTrainAlgoArgs
from src.specific.dt.trainer.pe_dt_data_trainer import DtPeDataTrainer
from src.specific.dt.trainer.pe_dt_io_trainer import DtPeIoTrainer
from src.specific.dt.trainer.pe_dt_train_writer import DtPeTrainWriter
from src.specific.dt.trainer.pe_dt_train_report_args import DtPeTrainReportArgs
from src.specific.dt.trainer.pe_dt_logic_trainer import DtPeLogicTrainer


class TestDtPeIoTrainer(unittest.TestCase):

    def setUp(self):
        self.reader = create_autospec(DtPeDataTrainer, instance=True)
        self.trainer = create_autospec(DtPeLogicTrainer, instance=True)
        self.writer = create_autospec(DtPeTrainWriter, instance=True)

    # ---------- __init__ ----------

    def test_init_sets_dependencies(self):
        io_trainer = DtPeIoTrainer(
            reader=self.reader,
            trainer=self.trainer,
            writer=self.writer,
        )

        self.assertIs(self.reader, io_trainer.reader)
        self.assertIs(self.trainer, io_trainer.trainer)
        self.assertIs(self.writer, io_trainer.writer)

    def test_init_raises_when_reader_is_none(self):
        with self.assertRaises(ValueError) as context:
            DtPeIoTrainer(
                reader=None,
                trainer=self.trainer,
                writer=self.writer,
            )

        self.assertEqual("reader cannot be None", str(context.exception))

    def test_init_raises_when_trainer_is_none(self):
        with self.assertRaises(ValueError) as context:
            DtPeIoTrainer(
                reader=self.reader,
                trainer=None,
                writer=self.writer,
            )

        self.assertEqual("trainer cannot be None", str(context.exception))

    def test_init_raises_when_writer_is_none(self):
        with self.assertRaises(ValueError) as context:
            DtPeIoTrainer(
                reader=self.reader,
                trainer=self.trainer,
                writer=None,
            )

        self.assertEqual("writer cannot be None", str(context.exception))

    def test_init_raises_when_reader_has_wrong_type(self):
        with self.assertRaises(TypeError) as context:
            DtPeIoTrainer(
                reader="not a reader",
                trainer=self.trainer,
                writer=self.writer,
            )

        self.assertEqual(
            "reader must be of type DtPeDataTrainer, but got str",
            str(context.exception),
        )

    def test_init_raises_when_trainer_has_wrong_type(self):
        with self.assertRaises(TypeError) as context:
            DtPeIoTrainer(
                reader=self.reader,
                trainer="not a trainer",
                writer=self.writer,
            )

        self.assertEqual(
            "trainer must be of type DtPeLogicTrainer, but got str",
            str(context.exception),
        )

    def test_init_raises_when_writer_has_wrong_type(self):
        with self.assertRaises(TypeError) as context:
            DtPeIoTrainer(
                reader=self.reader,
                trainer=self.trainer,
                writer="not a writer",
            )

        self.assertEqual(
            "writer must be of type DtPeTrainWriter, but got str",
            str(context.exception),
        )

    # ---------- train ----------

    def test_train_reads_trains_and_writes_output(self):
        io_trainer = DtPeIoTrainer(
            reader=self.reader,
            trainer=self.trainer,
            writer=self.writer,
        )

        algo_args = create_autospec(DtPeTrainAlgoArgs, instance=True)
        report_args = create_autospec(DtPeTrainReportArgs, instance=True)

        data = object()
