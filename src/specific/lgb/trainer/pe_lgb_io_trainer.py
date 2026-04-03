from src.common.validator.args_validator import ArgsValidator
from src.specific.lgb.trainer.pe_lgb_data_trainer import LgbPeDataTrainer
from src.specific.lgb.trainer.pe_lgb_train_algo_args import LgbPeTrainAlgoArgs
from src.specific.lgb.trainer.pe_lgb_train_report_args import LgbPeTrainReportArgs
from src.specific.lgb.trainer.pe_lgb_logic_trainer import LgbPeLogicTrainer
from src.specific.lgb.trainer.pe_lgb_train_writer import LgbPeTrainWriter


class LgbPeIoTrainer:
    """File I/O operations for training workflow."""

    def __init__(self, reader: LgbPeDataTrainer, trainer: LgbPeLogicTrainer, writer: LgbPeTrainWriter):
        ArgsValidator.require_type_not_none(reader, LgbPeDataTrainer, "reader")
        ArgsValidator.require_type_not_none(trainer, LgbPeLogicTrainer, "trainer")
        ArgsValidator.require_type_not_none(writer, LgbPeTrainWriter, "writer")
        self.reader = reader
        self.trainer = trainer
        self.writer = writer

    def train(self, algo_args: LgbPeTrainAlgoArgs, report_args: LgbPeTrainReportArgs):
        """Orchestrate training: read data, train model, write outputs."""
        data = self.reader.read_csv_to_df(report_args)
        result = self.trainer.train(algo_args, data)
        self.writer.write_output(report_args, result)
