import os

from src.common.validator import FileValidator
from tests.integration.rf.pe_rf_flow_state import RfPeFlowTestState


class RfPeTestStateProvider:

    @staticmethod
    def get_state(data_dir):
        input_dir = os.path.join(data_dir, 'integration', 'rf', 'input')
        input_csv_file = os.path.join(input_dir, 'malware_input_full_data.csv')
        output_dir_path = os.path.join(data_dir, 'integration', 'rf', 'output')
        return RfPeTestStateProvider.get_state_by_input_file_and_output_dir(input_csv_file, output_dir_path)

    @staticmethod
    def get_state_by_input_file_and_output_dir(input_csv_file, output_dir_path):
        FileValidator.validate_file(input_csv_file, True, False)
        FileValidator.validate_directory(output_dir_path, True, True)

        state = RfPeFlowTestState()
        state.input_csv_file = input_csv_file
        state.output_root_dir_path = output_dir_path

        state.output_preprocess_dir_path = os.path.join(state.output_root_dir_path, 'preprocess')
        state.output_clean_csv_file = os.path.join(state.output_preprocess_dir_path, 'clean.csv')
        state.output_preprocess_csv_file = os.path.join(state.output_preprocess_dir_path, 'preprocess.csv')

        state.output_train_dir_path = os.path.join(state.output_root_dir_path, 'train')
        state.output_model_file = os.path.join(state.output_train_dir_path, 'random_forest_model.joblib')

        state.output_evaluate_dir_path = os.path.join(state.output_root_dir_path, 'evaluate')
        state.rf_test_metrics_json_file = os.path.join(state.output_evaluate_dir_path, 'test_metrics.json')

        return state
