from src.common.validator.args_validator import ArgsValidator
from src.specific.xgb.trainer.pe_xgb_data_trainer import XgbPeDataTrainer
from src.specific.xgb.trainer.pe_xgb_train_algo_args import XgbPeTrainAlgoArgs
from src.specific.xgb.trainer.pe_xgb_train_report_args import XgbPeTrainReportArgs
from src.specific.xgb.trainer.pe_xgb_logic_trainer import XgbPeLogicTrainer
from src.specific.xgb.trainer.pe_xgb_train_writer import XgbPeTrainWriter


class XgbPeIoTrainer:
    """File I/O operations for training workflow."""

    def __init__(self, reader: XgbPeDataTrainer, trainer: XgbPeLogicTrainer, writer: XgbPeTrainWriter):
        ArgsValidator.require_type_not_none(reader, XgbPeDataTrainer, "reader")
        ArgsValidator.require_type_not_none(trainer, XgbPeLogicTrainer, "trainer")
        ArgsValidator.require_type_not_none(writer, XgbPeTrainWriter, "writer")
        self.reader = reader
        self.trainer = trainer
        self.writer = writer

    def train(self, algo_args: XgbPeTrainAlgoArgs, report_args: XgbPeTrainReportArgs):
        """Orchestrate training: read data, train model, write outputs."""
        data = self.reader.read_csv_to_df(report_args)
        result = self.trainer.train(algo_args, data)
        self.writer.write_output(report_args, result)
