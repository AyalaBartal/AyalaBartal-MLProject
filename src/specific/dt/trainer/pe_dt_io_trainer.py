import pandas as pd

from src.specific.dt.trainer import DtPeTrainAlgoArgs, DtPeTrainReportArgs
from src.specific.dt.trainer.pe_dt_trainer import DtPeDataTrainer
from src.specific.dt.trainer.pe_dt_train_writer import DtPeTrainWriter


class DtPeIoTrainer:

    def __init__(self, trainer: DtPeDataTrainer, writer: DtPeTrainWriter):
        self.trainer = trainer
        self.writer = writer

    def train(self, algo_args: DtPeTrainAlgoArgs, report_args: DtPeTrainReportArgs):
        data = pd.read_csv(report_args.input_csv)
        result = self.trainer.train(algo_args, data)
        self.writer.write_output(report_args, result)
