import unittest
from unittest.mock import create_autospec, Mock
import pandas as pd

from src.specific.dt.trainer.pe_dt_report_trainer import DtPeReportTrainer
from src.specific.dt.trainer.pe_dt_data_trainer import DtPeDataTrainer
from src.specific.dt.trainer.pe_dt_model_trainer import DtPeModelTrainer
from src.specific.dt.trainer.pe_dt_logic_trainer import DtPeLogicTrainer
from src.specific.dt.trainer.pe_dt_train_algo_args import DtPeTrainAlgoArgs


class TestDtPeLogicTrainer(unittest.TestCase):

    def setUp(self):
        self.reader = create_autospec(DtPeDataTrainer, instance=True)
        self.trainer = create_autospec(DtPeModelTrainer, instance=True)
        self.reporter = create_autospec(DtPeReportTrainer, instance=True)

        self.logic_trainer = DtPeLogicTrainer(
            reader=self.reader,
            trainer=self.trainer,
            reporter=self.reporter,
        )

    def test_init_sets_dependencies(self):
        self.assertIs(self.reader, self.logic_trainer.data_reader)
        self.assertIs(self.trainer, self.logic_trainer.trainer)
        self.assertIs(self.reporter, self.logic_trainer.reporter)

    def test_train_runs_full_pipeline_and_returns_report(self):
        args = Mock(spec=DtPeTrainAlgoArgs)
        args.label = "is_malware"

        data = pd.DataFrame({
            "f1": [1, 2, 3],
            "f2": [10, 20, 30],
            "is_malware": [0, 1, 0],
        })

        ml_label = pd.Series([0, 1, 0], name="is_malware")
        ml_features = pd.DataFrame({
            "f1": [1, 2, 3],
            "f2": [10, 20, 30],
        })

        model_before_fit = Mock(name="model_before_fit")
        model_after_fit = Mock(name="model_after_fit")
        skf = Mock(name="skf")
        scores = {"test_accuracy": [0.9], "test_roc_auc": [0.8]}
        con_matrix = [[2, 0], [0, 1]]
        expected_report = {"summary": "ok"}

        self.reader.get_label_as_series.return_value = ml_label
        self.reader.get_features_data_frame.return_value = ml_features
        self.trainer.get_decision_tree_classifier.return_value = model_before_fit
        self.trainer.get_split_train_test.return_value = skf
        self.trainer.get_cross_validate_score.return_value = scores
        self.trainer.fit_model.return_value = model_after_fit
        self.reporter.get_confusion_matrix.return_value = con_matrix
        self.reporter.get_report.return_value = expected_report

        result = self.logic_trainer.train(args, data)

        self.assertEqual(expected_report, result)

        self.reader.get_label_as_series.assert_called_once_with(data, "is_malware")
        self.reader.get_features_data_frame.assert_called_once_with(data, "is_malware")

        self.trainer.get_decision_tree_classifier.assert_called_once_with(args)
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

    def test_train_calls_methods_in_expected_order(self):
        args = Mock(spec=DtPeTrainAlgoArgs)
        args.label = "label"
        data = pd.DataFrame({"x": [1], "label": [0]})

        ml_label = pd.Series([0], name="label")
        ml_features = pd.DataFrame({"x": [1]})
        model_before_fit = Mock(name="model_before_fit")
        model_after_fit = Mock(name="model_after_fit")
        skf = Mock(name="skf")
        scores = Mock(name="scores")
        con_matrix = Mock(name="con_matrix")
        report = Mock(name="report")

        call_log = []

        def log_and_return(name, value):
            def _fn(*args_, **kwargs_):
                call_log.append(name)
                return value
            return _fn

        self.reader.get_label_as_series.side_effect = log_and_return("get_label_as_series", ml_label)
        self.reader.get_features_data_frame.side_effect = log_and_return("get_features_data_frame", ml_features)
        self.trainer.get_decision_tree_classifier.side_effect = log_and_return("get_decision_tree_classifier", model_before_fit)
        self.trainer.get_split_train_test.side_effect = log_and_return("get_split_train_test", skf)
        self.trainer.get_cross_validate_score.side_effect = log_and_return("get_cross_validate_score", scores)
        self.trainer.fit_model.side_effect = log_and_return("fit_model", model_after_fit)
        self.reporter.get_confusion_matrix.side_effect = log_and_return("get_confusion_matrix", con_matrix)
        self.reporter.get_report.side_effect = log_and_return("get_report", report)

        result = self.logic_trainer.train(args, data)

        self.assertIs(report, result)
        self.assertEqual(
            [
                "get_label_as_series",
                "get_features_data_frame",
                "get_decision_tree_classifier",
                "get_split_train_test",
                "get_cross_validate_score",
                "fit_model",
                "get_confusion_matrix",
                "get_report",
            ],
            call_log,
        )

    def test_init_raises_when_reader_is_none(self):
        with self.assertRaises(ValueError) as context:
            DtPeLogicTrainer(None, self.trainer, self.reporter)

        self.assertEqual("reader cannot be None", str(context.exception))

    def test_init_raises_when_trainer_is_none(self):
        with self.assertRaises(ValueError) as context:
            DtPeLogicTrainer(self.reader, None, self.reporter)

        self.assertEqual("trainer cannot be None", str(context.exception))

    def test_init_raises_when_reporter_is_none(self):
        with self.assertRaises(ValueError) as context:
            DtPeLogicTrainer(self.reader, self.trainer, None)

        self.assertEqual("reporter cannot be None", str(context.exception))

    def test_init_raises_when_reader_has_wrong_type(self):
        with self.assertRaises(TypeError) as context:
            DtPeLogicTrainer(object(), self.trainer, self.reporter)

        self.assertEqual(
            "reader must be of type DtPeDataTrainer, but got object",
            str(context.exception),
        )

    def test_init_raises_when_trainer_has_wrong_type(self):
        with self.assertRaises(TypeError) as context:
            DtPeLogicTrainer(self.reader, object(), self.reporter)

        self.assertEqual(
            "trainer must be of type DtPeModelTrainer, but got object",
            str(context.exception),
        )

    def test_init_raises_when_reporter_has_wrong_type(self):
        with self.assertRaises(TypeError) as context:
            DtPeLogicTrainer(self.reader, self.trainer, object())

        self.assertEqual(
            "reporter must be of type DtPeReportTrainer, but got object",
            str(context.exception),
        )
