import os

import pandas as pd

from src.specific.dt.trainer import DtPeDataTrainer


class DtPeIoTrainer:

    def __init__(self, trainer: DtPeDataTrainer):
        self.trainer = trainer

    def train(self, args):
        self.make_output_dirs(args.out_model, args.out_report_json)
        data = self.get_data_from_csv(args)
        self.trainer.train(args, data)

    def make_output_dirs(self, out_model, out_report_json):
        os.makedirs(os.path.dirname(out_model), exist_ok=True)
        os.makedirs(os.path.dirname(out_report_json), exist_ok=True)

    def get_data_from_csv(self, args):
        return pd.read_csv(args.input_csv)
