from src.common.validator.args_validator import ArgsValidator
from src.specific.rf.trainer.pe_rf_data_trainer import RfPeDataTrainer
from src.specific.rf.trainer.pe_rf_train_algo_args import RfPeTrainAlgoArgs
from src.specific.rf.trainer.pe_rf_train_report_args import RfPeTrainReportArgs
from src.specific.rf.trainer.pe_rf_logic_trainer import RfPeLogicTrainer
from src.specific.rf.trainer.pe_rf_train_writer import RfPeTrainWriter


class RfPeIoTrainer:

    def __init__(self, reader: RfPeDataTrainer, trainer: RfPeLogicTrainer, writer: RfPeTrainWriter):
        ArgsValidator.require_type_not_none(reader, RfPeDataTrainer, "reader")
        ArgsValidator.require_type_not_none(trainer, RfPeLogicTrainer, "trainer")
        ArgsValidator.require_type_not_none(writer, RfPeTrainWriter, "writer")
        self.reader = reader
        self.trainer = trainer
        self.writer = writer

    def train(self, algo_args: RfPeTrainAlgoArgs, report_args: RfPeTrainReportArgs):
        data = self.reader.read_csv_to_df(report_args)
        result = self.trainer.train(algo_args, data)
        self.writer.write_output(report_args, result)
