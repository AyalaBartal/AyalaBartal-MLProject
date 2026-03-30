import unittest
import time

import pandas as pd

from src.common.csv.feature_index_provider import CsvBrazilianProvider
from src.common.csv.csv_io_validate import CsvIoValidate
from src.common.precleanup.pe_preprocess_cleaner import PePreprocessCleaner
from src.specific.lr.evaluate.pe_lr_evaluator_provider import LrPeEvaluatorProvider
from src.specific.lr.evaluate.pe_lr_evaluate_input_args import LrPeEvaluateInputArgs
from src.specific.lr.evaluate.pe_lr_evaluate_algo_args import LrPeEvaluateAlgoArgs
from src.specific.lr.evaluate.pe_lr_evaluate_output_args import LrPeEvaluateOutputArgs
from src.common.preprocessor import LrPeDataPreprocessCsvArgs
from src.common.preprocessor import LrPePreprocessorProvider
from src.common.preprocessor import LrPeCsvPreprocessMapper
from src.specific.lr.trainer.pe_lr_train_algo_args import LrPeTrainAlgoArgs
from src.specific.lr.trainer.pe_lr_train_args import LrPeTrainReportArgs
from src.specific.lr.trainer.pe_lr_trainer_provider import LrPeTrainerProvider
from tests.integration.lr.pe_lr_state_provider import LrPeTestStateProvider
from tests.utils.paths_provider import PathsProvider

"""
Ordered integration test for ML logistic-regression flow.

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
class TestLrPeIntegration(unittest.TestCase):

    start_time = 0
    state = None
    columns_provider = None

    @classmethod
    def setUpClass(cls):
        cls.start_time = time.perf_counter()
        data_dir = PathsProvider.get_test_data_dir()
        cls.state = LrPeTestStateProvider.get_state(data_dir)

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
        args = LrPeDataPreprocessCsvArgs(self.state.output_clean_csv_file, self.state.output_preprocess_csv_file)
        lr_pe_csv_preprocessor = LrPeCsvPreprocessMapper(LrPePreprocessorProvider.get_mapper())
        lr_pe_csv_preprocessor.map(args)
        print("End success test_03_preprocessor")

    def test_04_trainer(self):
        print("Start test_04_trainer")
        algo_args = LrPeTrainAlgoArgs()
        report_args = LrPeTrainReportArgs(self.state.output_preprocess_csv_file, self.state.output_train_dir_path)
        io_trainer = LrPeTrainerProvider.get_io_trainer()
        io_trainer.train(algo_args, report_args)
        print("End success test_04_trainer")

    def test_05_evaluate(self):
        print("Start test_05_evaluate")
        input_args = LrPeEvaluateInputArgs(self.state.output_train_dir_path)
        # The input csv is not the original input but the input after clean and reprocess.
        input_args.input_csv = self.state.output_preprocess_csv_file
        # The model from the training
        input_args.input_model = self.state.output_model_file

        algo_args = LrPeEvaluateAlgoArgs()
        output_args = LrPeEvaluateOutputArgs(self.state.output_evaluate_dir_path)

        provider = LrPeEvaluatorProvider()
        evaluator = provider.get_evaluator()

        evaluator.evaluate(input_args, algo_args, output_args)
        print("End success test_05_evaluate")


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(TestLrPeIntegration("test_01_input_validation"))
    test_suite.addTest(TestLrPeIntegration("test_02_input_cleanup"))
    test_suite.addTest(TestLrPeIntegration("test_03_preprocessor"))
    test_suite.addTest(TestLrPeIntegration("test_04_trainer"))
    test_suite.addTest(TestLrPeIntegration("test_05_evaluate"))
    return test_suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2, failfast=True)
    result = runner.run(suite())

    if result.wasSuccessful():
        print("Success suite: all integration steps completed successfully.")
    else:
        print("Fail suite: integration flow stopped because at least one step failed.")
