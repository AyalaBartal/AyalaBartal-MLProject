import os

import pandas as pd

from src.specific.dt.trainer.pe_dt_trainer import DtPeDataTrainer
from src.specific.dt.trainer.pe_dt_train_writer import DtPeTrainWriter


class DtPeIoTrainer:

    def __init__(self, trainer: DtPeDataTrainer, writer: DtPeTrainWriter):
        self.trainer = trainer
        self.writer = writer

    def train(self, args):
        self.make_output_dirs(args.out_model, args.out_report_json)
        data = self.get_data_from_csv(args)

        result = self.trainer.train(args, data)

        self.writer.write_output(result)

    def make_output_dirs(self, out_model, out_report_json):
        os.makedirs(os.path.dirname(out_model), exist_ok=True)
        os.makedirs(os.path.dirname(out_report_json), exist_ok=True)

    def get_data_from_csv(self, args):
        return pd.read_csv(args.input_csv)
