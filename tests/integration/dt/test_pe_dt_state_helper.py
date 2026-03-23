import os

from src.common.validator import FileValidator
from tests.integration.dt.test_pe_dt_flow_state import TestPePdFlowState


class TestPePdStateHelper:

    @staticmethod
    def get_state(data_dir):
        input_dir = os.path.join(data_dir, 'integration', 'dt', 'input')
        input_csv_file = os.path.join(input_dir, 'malware_input_full_data.csv')
        output_dir_path = os.path.join(data_dir, 'integration', 'dt', 'output')
        return TestPePdStateHelper.get_state_by_input_file_and_output_dir(input_csv_file, output_dir_path)

    @staticmethod
    def get_state_by_input_file_and_output_dir(input_csv_file, output_dir_path):
        FileValidator.validate_file(input_csv_file, True, False)
        FileValidator.validate_directory(output_dir_path, True, True)

        state = TestPePdFlowState()
        state.input_csv_file = input_csv_file
        state.output_root_dir_path = output_dir_path

        state.output_preprocess_dir_path = os.path.join(state.output_root_dir_path, 'preprocess')
        state.output_clean_csv_file = os.path.join(state.output_preprocess_dir_path, 'clean.csv')
        state.output_preprocess_csv_file = os.path.join(state.output_preprocess_dir_path, 'preprocess.csv')

        state.output_train_dir_path = os.path.join(state.output_root_dir_path, 'train')
        state.output_model_file = os.path.join(state.output_train_dir_path, 'decision_tree_model.joblib')

        state.output_evaluate_dir_path = os.path.join(state.output_root_dir_path, 'evaluate')
        state.dt_test_metrics_json_file = os.path.join(state.output_evaluate_dir_path, 'test_metrics.json')

        return state
