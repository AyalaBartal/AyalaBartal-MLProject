import pandas as pd

from src.specific.dt.trainer.pe_dt_trainer import DtPeDataTrainer
from src.specific.dt.trainer.pe_dt_train_writer import DtPeTrainWriter


class DtPeIoTrainer:

    def __init__(self, trainer: DtPeDataTrainer, writer: DtPeTrainWriter):
        self.trainer = trainer
        self.writer = writer

    def train(self, args):
        data = pd.read_csv(args.input_csv)
        result = self.trainer.train(args, data)
        self.writer.write_output(result)
