from src.specific.lr.trainer.pe_lr_train_algo_args import LrPeTrainAlgoArgs


class LrPeTrainArgs:

    def __init__(self, input_csv, output_dir, algo_args: LrPeTrainAlgoArgs = None):
        self.input_csv = input_csv
        self.output_dir = output_dir
        self.algo_args = algo_args if algo_args else LrPeTrainAlgoArgs()
