from src.common.validator.args_validator import ArgsValidator
from src.specific.cbst.trainer.pe_cbst_data_trainer import CbstPeDataTrainer
from src.specific.cbst.trainer.pe_cbst_train_algo_args import CbstPeTrainAlgoArgs
from src.specific.cbst.trainer.pe_cbst_train_report_args import CbstPeTrainReportArgs
from src.specific.cbst.trainer.pe_cbst_logic_trainer import CbstPeLogicTrainer
from src.specific.cbst.trainer.pe_cbst_train_writer import CbstPeTrainWriter


class CbstPeIoTrainer:
    """File I/O operations for training workflow."""

    def __init__(self, reader: CbstPeDataTrainer, trainer: CbstPeLogicTrainer, writer: CbstPeTrainWriter):
        ArgsValidator.require_type_not_none(reader, CbstPeDataTrainer, "reader")
        ArgsValidator.require_type_not_none(trainer, CbstPeLogicTrainer, "trainer")
        ArgsValidator.require_type_not_none(writer, CbstPeTrainWriter, "writer")
        self.reader = reader
        self.trainer = trainer
        self.writer = writer

    def train(self, algo_args: CbstPeTrainAlgoArgs, report_args: CbstPeTrainReportArgs):
        """Orchestrate training: read data, train model, write outputs."""
        data = self.reader.read_csv_to_df(report_args)
        result = self.trainer.train(algo_args, data)
        self.writer.write_output(report_args, result)
