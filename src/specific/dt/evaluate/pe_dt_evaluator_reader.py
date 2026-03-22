import pandas as pd
from joblib import load

from src.specific.dt.evaluate.file_util import FileUtil


class DtPeEvaluatorReader:

    def __init__(self, validator: FileUtil):
        self.validator = validator

    def validate_output(self, args_out):
        self.validator.validate_directory(args_out.output_dir, True, True)

    def validate_input(self, args_in):
        self.validator.validate_directory(args_in.input_dir, True, False)

    def read_dt_model_from_joblib_file(self, args_in):
        return load(args_in.input_model)

    def read_csv_to_df(self, input_csv):
        return pd.read_csv(input_csv)
