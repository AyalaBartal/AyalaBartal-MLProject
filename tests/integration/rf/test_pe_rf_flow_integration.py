import unittest
import time

import pandas as pd

from src.common.csv.feature_index_provider import CsvBrazilianProvider
from src.common.csv.csv_io_validate import CsvIoValidate
from src.common.precleanup.pe_preprocess_cleaner import PePreprocessCleaner
from src.specific.rf.evaluate.pe_rf_evaluator_provider import RfPeEvaluatorProvider
from src.specific.rf.evaluate.pe_rf_evaluate_input_args import RfPeEvaluateInputArgs
from src.specific.rf.evaluate.pe_rf_evaluate_algo_args import RfPeEvaluateAlgoArgs
from src.specific.rf.evaluate.pe_rf_evaluate_output_args import RfPeEvaluateOutputArgs
from src.common.preprocessor import RfPeDataPreprocessCsvArgs
from src.common.preprocessor import RfPePreprocessorProvider
from src.common.preprocessor import RfPeCsvPreprocessMapper
from src.specific.rf.trainer.pe_rf_train_algo_args import RfPeTrainAlgoArgs
from src.specific.rf.trainer.pe_rf_train_args import RfPeTrainReportArgs
from src.specific.rf.trainer.pe_rf_trainer_provider import RfPeTrainerProvider
from tests.integration.rf.pe_rf_state_provider import RfPeTestStateProvider
from tests.utils.paths_provider import PathsProvider

"""
Ordered integration test for ML random-forest flow.

Flow:
1. input validation
2. input cleanup (filter out invalid rows)
3. preprocessor
4. trainer
5. evaluate

Notes:
- State is stored in a separate test data holder class.
- Each step writes a success marker for the next step.
- If a step fails, the suite stops because the custom runner uses failfast=True.
- Each step includes:
    * clear failure messages
    * clear success messages
"""


@unittest.skip("Skip the entire test class until the data input/output and environment variable issues are resolved")
class TestRfPeIntegration(unittest.TestCase):

    start_time = 0
    state = None
    columns_provider = None

    @classmethod
    def setUpClass(cls):
        cls.start_time = time.perf_counter()
        data_dir = PathsProvider.get_test_data_dir()
        cls.state = RfPeTestStateProvider.get_state(data_dir)

        cls.columns_provider = CsvBrazilianProvider()
        print("Start test {} with state {}".format(cls.__class__.__name__, cls.state))

    @classmethod
    def tearDownClass(cls):
        duration = time.perf_counter() - cls.start_time
        seconds = f"{duration % 60:.0f}"
        print("End test {} took {} seconds".format(cls.__class__.__name__, seconds))

    def test_01_input_validation(self):
        print("Start test_01_input_validation")
        headers = self.columns_provider.get_map_header_by_index()
        # Define max file size as 500MB
        max_size_bytes = 500 * 1024 * 1024
        CsvIoValidate.validate_csv_file(self.state.input_csv_file, headers, max_size_bytes)
        print("End success test_01_input_validation")

    def test_02_input_cleanup(self):
        print("Start test_02_clean_input")
        pe_preprocess_cleaner = PePreprocessCleaner(self.columns_provider)
        input_data = pd.read_csv(self.state.input_csv_file)
        output_data = pe_preprocess_cleaner.clean(input_data)
        output_data.to_csv(self.state.output_clean_csv_file)
        print("End success test_02_clean_input")

    def test_03_preprocessor(self):
        print("Start test_03_preprocessor")
        args = RfPeDataPreprocessCsvArgs(self.state.output_clean_csv_file, self.state.output_preprocess_csv_file)
        rf_pe_csv_preprocessor = RfPeCsvPreprocessMapper(RfPePreprocessorProvider.get_mapper())
        rf_pe_csv_preprocessor.map(args)
        print("End success test_03_preprocessor")

    def test_04_trainer(self):
        print("Start test_04_trainer")
        algo_args = RfPeTrainAlgoArgs()
        report_args = RfPeTrainReportArgs(self.state.output_preprocess_csv_file, self.state.output_train_dir_path)
        io_trainer = RfPeTrainerProvider.get_io_trainer()
        io_trainer.train(algo_args, report_args)
        print("End success test_04_trainer")

    def test_05_evaluate(self):
        print("Start test_05_evaluate")
        input_args = RfPeEvaluateInputArgs(self.state.output_train_dir_path)
        # The input csv is not the original input but the input after clean and reprocess.
        input_args.input_csv = self.state.output_preprocess_csv_file
        # The model from the training
        input_args.input_model = self.state.output_model_file

        algo_args = RfPeEvaluateAlgoArgs()
        output_args = RfPeEvaluateOutputArgs(self.state.output_evaluate_dir_path)

        provider = RfPeEvaluatorProvider()
        evaluator = provider.get_evaluator()

        evaluator.evaluate(input_args, algo_args, output_args)
        print("End success test_05_evaluate")


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(TestRfPeIntegration("test_01_input_validation"))
    test_suite.addTest(TestRfPeIntegration("test_02_input_cleanup"))
    test_suite.addTest(TestRfPeIntegration("test_03_preprocessor"))
    test_suite.addTest(TestRfPeIntegration("test_04_trainer"))
    test_suite.addTest(TestRfPeIntegration("test_05_evaluate"))
    return test_suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2, failfast=True)
    result = runner.run(suite())

    if result.wasSuccessful():
        print("Success suite: all integration steps completed successfully.")
    else:
        print("Fail suite: integration flow stopped because at least one step failed.")
