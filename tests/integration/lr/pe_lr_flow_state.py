import os

from src.common.validator import FileValidator


class LrPeFlowTestState:

    def __init__(self):
        self.input_csv_file = None
        self.output_root_dir_path = None
        self.output_preprocess_dir_path = None
        self.output_clean_csv_file = None
        self.output_preprocess_csv_file = None
        self.output_train_dir_path = None
        self.output_model_file = None
        self.output_evaluate_dir_path = None
        self.lr_test_metrics_json_file = None
