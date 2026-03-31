from src.specific.ml.trainer.pe_ml_train_algo_args import MlPeTrainAlgoArgs


class MlPeTrainArgs:

    def __init__(self, input_csv, output_dir, algo_args: MlPeTrainAlgoArgs = None):
        self.input_csv = input_csv
        self.output_dir = output_dir
        self.algo_args = algo_args if algo_args else MlPeTrainAlgoArgs()
