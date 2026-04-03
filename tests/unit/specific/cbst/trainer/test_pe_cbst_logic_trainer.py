import unittest
from unittest.mock import create_autospec, Mock, patch
import pandas as pd

from src.specific.lgb.trainer.pe_lgb_report_trainer import LgbPeReportTrainer
from src.specific.lgb.trainer.pe_lgb_data_trainer import LgbPeDataTrainer
from src.specific.lgb.trainer.pe_lgb_model_trainer import LgbPeModelTrainer
from src.specific.lgb.trainer.pe_lgb_logic_trainer import LgbPeLogicTrainer
from src.specific.lgb.trainer.pe_lgb_train_algo_args import LgbPeTrainAlgoArgs


class TestCbstPeLogicTrainer(unittest.TestCase):

    def setUp(self):
        self.reader = create_autospec(LgbPeDataTrainer, instance=True)
        self.trainer = create_autospec(LgbPeModelTrainer, instance=True)
        self.reporter = create_autospec(LgbPeReportTrainer, instance=True)

        self.logic_trainer = LgbPeLogicTrainer(
            reader=self.reader,
            trainer=self.trainer,
            reporter=self.reporter,
        )

    def test_init_sets_dependencies(self):
        self.assertIs(self.reader, self.logic_trainer.data_reader)
        self.assertIs(self.trainer, self.logic_trainer.trainer)
        self.assertIs(self.reporter, self.logic_trainer.reporter)

    @patch('src.specific.lgb.trainer.pe_lgb_logic_trainer.DtPePreprocessorProvider')
    def test_train_runs_full_pipeline_and_returns_report(self, mock_provider):
        args = Mock(spec=LgbPeTrainAlgoArgs)
        args.label = "is_malware"

        raw_data = pd.DataFrame({
            "f1": [1, 2, 3],
            "f2": [10, 20, 30],
            "is_malware": [0, 1, 0],
        })

        preprocessed_data = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0],
            "f2": [10.0, 20.0, 30.0],
            "is_malware": [0, 1, 0],
        })

        ml_label = pd.Series([0, 1, 0], name="is_malware")
        ml_features = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0],
            "f2": [10.0, 20.0, 30.0],
        })

        model_before_fit = Mock(name="model_before_fit")
        model_after_fit = Mock(name="model_after_fit")
        skf = Mock(name="skf")
        scores = {"test_accuracy": [0.9], "test_roc_auc": [0.8]}
        con_matrix = [[2, 0], [0, 1]]
        expected_report = {"summary": "ok"}

        mock_mapper = Mock()
        mock_mapper.map.return_value = preprocessed_data
        mock_provider.get_mapper.return_value = mock_mapper

        self.reader.get_label_as_series.return_value = ml_label
        self.reader.get_features_data_frame.return_value = ml_features
        self.trainer.get_lightgbm_classifier.return_value = model_before_fit
        self.trainer.get_split_train_test.return_value = skf
        self.trainer.get_cross_validate_score.return_value = scores
        self.trainer.fit_model.return_value = model_after_fit
        self.reporter.get_confusion_matrix.return_value = con_matrix
        self.reporter.get_report.return_value = expected_report

        result = self.logic_trainer.train(args, raw_data)

        self.assertEqual(expected_report, result)

        self.trainer.get_lightgbm_classifier.assert_called_once_with(args)
        self.trainer.get_split_train_test.assert_called_once_with(args)
        self.trainer.get_cross_validate_score.assert_called_once_with(
            model_before_fit, skf, ml_features, ml_label
        )
        self.trainer.fit_model.assert_called_once_with(
            model_before_fit, ml_features, ml_label
        )

        self.reporter.get_confusion_matrix.assert_called_once_with(
            model_after_fit, skf, ml_features, ml_label
        )
        self.reporter.get_report.assert_called_once_with(
            args, ml_features, model_after_fit, con_matrix, scores
        )

    @patch('src.specific.lgb.trainer.pe_lgb_logic_trainer.DtPePreprocessorProvider')
    def test_train_calls_preprocessor(self, mock_provider):
        args = Mock(spec=LgbPeTrainAlgoArgs)
        args.label = "label"

        raw_data = pd.DataFrame({"x": [1], "label": [0]})
        preprocessed_data = pd.DataFrame({"x": [1.0], "label": [0]})

        ml_label = pd.Series([0], name="label")
        ml_features = pd.DataFrame({"x": [1.0]})
        model = Mock(name="model")
        skf = Mock(name="skf")
        scores = Mock(name="scores")
        con_matrix = Mock(name="con_matrix")
        report = Mock(name="report")

        mock_mapper = Mock()
        mock_mapper.map.return_value = preprocessed_data
        mock_provider.get_mapper.return_value = mock_mapper

        self.reader.get_label_as_series.return_value = ml_label
        self.reader.get_features_data_frame.return_value = ml_features
        self.trainer.get_lightgbm_classifier.return_value = model
        self.trainer.get_split_train_test.return_value = skf
        self.trainer.get_cross_validate_score.return_value = scores
        self.trainer.fit_model.return_value = model
        self.reporter.get_confusion_matrix.return_value = con_matrix
        self.reporter.get_report.return_value = report

        self.logic_trainer.train(args, raw_data)

        mock_mapper.map.assert_called_once_with(raw_data)

    @patch('src.specific.lgb.trainer.pe_lgb_logic_trainer.DtPePreprocessorProvider')
    def test_train_processes_data_in_correct_order(self, mock_provider):
        args = Mock(spec=LgbPeTrainAlgoArgs)
        args.label = "label"

        raw_data = pd.DataFrame({"x": [1, 2], "label": [0, 1]})
        preprocessed_data = pd.DataFrame({"x": [1.0, 2.0], "label": [0, 1]})

        ml_label = pd.Series([0, 1], name="label")
        ml_features = pd.DataFrame({"x": [1.0, 2.0]})
        model = Mock(name="model")
        skf = Mock(name="skf")
        scores = Mock(name="scores")
        con_matrix = Mock(name="con_matrix")
        report = Mock(name="report")

        mock_mapper = Mock()
        mock_mapper.map.return_value = preprocessed_data
        mock_provider.get_mapper.return_value = mock_mapper

        self.reader.get_label_as_series.return_value = ml_label
        self.reader.get_features_data_frame.return_value = ml_features
        self.trainer.get_lightgbm_classifier.return_value = model
        self.trainer.get_split_train_test.return_value = skf
        self.trainer.get_cross_validate_score.return_value = scores
        self.trainer.fit_model.return_value = model
        self.reporter.get_confusion_matrix.return_value = con_matrix
        self.reporter.get_report.return_value = report

        self.logic_trainer.train(args, raw_data)

        self.reader.get_label_as_series.assert_called_once()
        self.reader.get_features_data_frame.assert_called_once()

    def test_init_raises_when_reader_is_none(self):
        with self.assertRaises(ValueError) as context:
            LgbPeLogicTrainer(None, self.trainer, self.reporter)

        self.assertEqual("reader cannot be None", str(context.exception))

    def test_init_raises_when_trainer_is_none(self):
        with self.assertRaises(ValueError) as context:
            LgbPeLogicTrainer(self.reader, None, self.reporter)

        self.assertEqual("trainer cannot be None", str(context.exception))

    def test_init_raises_when_reporter_is_none(self):
        with self.assertRaises(ValueError) as context:
            LgbPeLogicTrainer(self.reader, self.trainer, None)

        self.assertEqual("reporter cannot be None", str(context.exception))

    def test_init_raises_when_reader_has_wrong_type(self):
        with self.assertRaises(TypeError) as context:
            LgbPeLogicTrainer(object(), self.trainer, self.reporter)

        self.assertEqual(
            "reader must be of type LgbPeDataTrainer, but got object",
            str(context.exception),
        )

    def test_init_raises_when_trainer_has_wrong_type(self):
        with self.assertRaises(TypeError) as context:
            LgbPeLogicTrainer(self.reader, object(), self.reporter)

        self.assertEqual(
            "trainer must be of type LgbPeModelTrainer, but got object",
            str(context.exception),
        )

    def test_init_raises_when_reporter_has_wrong_type(self):
        with self.assertRaises(TypeError) as context:
            LgbPeLogicTrainer(self.reader, self.trainer, object())

        self.assertEqual(
            "reporter must be of type LgbPeReportTrainer, but got object",
            str(context.exception),
        )


if __name__ == "__main__":
    unittest.main()
