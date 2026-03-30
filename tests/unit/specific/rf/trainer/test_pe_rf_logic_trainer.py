import unittest
from unittest.mock import create_autospec, Mock, patch
import pandas as pd

from src.specific.rf.trainer.pe_rf_report_trainer import RfPeReportTrainer
from src.specific.rf.trainer.pe_rf_data_trainer import RfPeDataTrainer
from src.specific.rf.trainer.pe_rf_model_trainer import RfPeModelTrainer
from src.specific.rf.trainer.pe_rf_logic_trainer import RfPeLogicTrainer
from src.specific.rf.trainer.pe_rf_train_algo_args import RfPeTrainAlgoArgs


class TestRfPeLogicTrainer(unittest.TestCase):

    def setUp(self):
        self.reader = create_autospec(RfPeDataTrainer, instance=True)
        self.trainer = create_autospec(RfPeModelTrainer, instance=True)
        self.reporter = create_autospec(RfPeReportTrainer, instance=True)

        self.logic_trainer = RfPeLogicTrainer(
            reader=self.reader,
            trainer=self.trainer,
            reporter=self.reporter,
        )

    def test_init_sets_dependencies(self):
        self.assertIs(self.reader, self.logic_trainer.data_reader)
        self.assertIs(self.trainer, self.logic_trainer.trainer)
        self.assertIs(self.reporter, self.logic_trainer.reporter)

    @patch('src.specific.rf.trainer.pe_rf_logic_trainer.DtPePreprocessorProvider')
    def test_train_runs_full_pipeline_and_returns_report(self, mock_provider):
        args = Mock(spec=RfPeTrainAlgoArgs)
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
        self.trainer.get_random_forest_classifier.return_value = model_before_fit
        self.trainer.get_split_train_test.return_value = skf
        self.trainer.get_cross_validate_score.return_value = scores
        self.trainer.fit_model.return_value = model_after_fit
        self.reporter.get_confusion_matrix.return_value = con_matrix
        self.reporter.get_report.return_value = expected_report

        result = self.logic_trainer.train(args, raw_data)

        self.assertEqual(expected_report, result)

        self.trainer.get_random_forest_classifier.assert_called_once_with(args)
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

    @patch('src.specific.rf.trainer.pe_rf_logic_trainer.DtPePreprocessorProvider')
    def test_train_calls_preprocessor(self, mock_provider):
        args = Mock(spec=RfPeTrainAlgoArgs)
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
        self.trainer.get_random_forest_classifier.return_value = model
        self.trainer.get_split_train_test.return_value = skf
        self.trainer.get_cross_validate_score.return_value = scores
        self.trainer.fit_model.return_value = model
        self.reporter.get_confusion_matrix.return_value = con_matrix
        self.reporter.get_report.return_value = report

        self.logic_trainer.train(args, raw_data)

        mock_mapper.map.assert_called_once_with(raw_data)

    def test_init_raises_when_reader_is_none(self):
        with self.assertRaises(ValueError) as context:
            RfPeLogicTrainer(None, self.trainer, self.reporter)

        self.assertEqual("reader cannot be None", str(context.exception))

    def test_init_raises_when_trainer_is_none(self):
        with self.assertRaises(ValueError) as context:
            RfPeLogicTrainer(self.reader, None, self.reporter)

        self.assertEqual("trainer cannot be None", str(context.exception))

    def test_init_raises_when_reporter_is_none(self):
        with self.assertRaises(ValueError) as context:
            RfPeLogicTrainer(self.reader, self.trainer, None)

        self.assertEqual("reporter cannot be None", str(context.exception))

    def test_init_raises_when_reader_has_wrong_type(self):
        with self.assertRaises(TypeError) as context:
            RfPeLogicTrainer(object(), self.trainer, self.reporter)

        self.assertEqual(
            "reader must be of type RfPeDataTrainer, but got object",
            str(context.exception),
        )

    def test_init_raises_when_trainer_has_wrong_type(self):
        with self.assertRaises(TypeError) as context:
            RfPeLogicTrainer(self.reader, object(), self.reporter)

        self.assertEqual(
            "trainer must be of type RfPeModelTrainer, but got object",
            str(context.exception),
        )

    def test_init_raises_when_reporter_has_wrong_type(self):
        with self.assertRaises(TypeError) as context:
            RfPeLogicTrainer(self.reader, self.trainer, object())

        self.assertEqual(
            "reporter must be of type RfPeReportTrainer, but got object",
            str(context.exception),
        )
