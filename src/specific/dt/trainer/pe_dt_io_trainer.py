from src.specific.dt.trainer import DtPeDataTrainer
from src.specific.dt.trainer.args_validator import ArgsValidator
from src.specific.dt.trainer.pe_dt_train_algo_args import DtPeTrainAlgoArgs
from src.specific.dt.trainer.pe_dt_train_report_args import DtPeTrainReportArgs
from src.specific.dt.trainer.pe_dt_logic_trainer import DtPeLogicTrainer
from src.specific.dt.trainer.pe_dt_train_writer import DtPeTrainWriter


class DtPeIoTrainer:

    def __init__(self, reader: DtPeDataTrainer, trainer: DtPeLogicTrainer, writer: DtPeTrainWriter):
        ArgsValidator.require_type_not_none(reader, DtPeDataTrainer, "reader")
        ArgsValidator.require_type_not_none(trainer, DtPeLogicTrainer, "trainer")
        ArgsValidator.require_type_not_none(writer, DtPeTrainWriter, "writer")
        self.reader = reader
        self.trainer = trainer
        self.writer = writer

    def train(self, algo_args: DtPeTrainAlgoArgs, report_args: DtPeTrainReportArgs):
        data = self.reader.read_csv_to_df(report_args)
        result = self.trainer.train(algo_args, data)
        self.writer.write_output(report_args, result)
