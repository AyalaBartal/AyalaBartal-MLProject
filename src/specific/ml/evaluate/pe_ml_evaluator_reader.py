import pandas as pd
import torch

from src.common.validator.file_validator import FileValidator


class MlPeEvaluatorReader:

    def __init__(self, validator: FileValidator):
        self.validator = validator

    def validate_output(self, args_out):
        self.validator.validate_directory(args_out.output_dir, True, True)

    def validate_input(self, args_in):
        self.validator.validate_directory(args_in.input_dir, True, False)

    def read_ml_model_from_pt_file(self, args_in):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.load(args_in.input_model, map_location=device)

    def read_csv_to_df(self, input_csv):
        return pd.read_csv(input_csv)
