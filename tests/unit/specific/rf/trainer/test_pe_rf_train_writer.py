import unittest
from unittest.mock import MagicMock

from src.specific.rf.trainer.pe_rf_train_writer import RfPeTrainWriter
from src.specific.rf.trainer.pe_rf_train_output_mapper import RfPeTrainOutputMapper
from src.specific.rf.trainer.pe_rf_train_output_writer import RfPeTrainOutputWriter
from src.specific.rf.trainer.pe_rf_train_result import RfPeTrainResult
from src.specific.rf.trainer.pe_rf_train_report_args import RfPeTrainReportArgs


class TestRfPeTrainWriter(unittest.TestCase):

    def setUp(self):
        self.mapper = object.__new__(RfPeTrainOutputMapper)
        self.writer = object.__new__(RfPeTrainOutputWriter)
        self.train_writer = RfPeTrainWriter(self.mapper, self.writer)

    def test_init_sets_dependencies(self):
        self.assertIs(self.mapper, self.train_writer.mapper)
        self.assertIs(self.writer, self.train_writer.writer)

    def test_init_raises_when_mapper_is_none(self):
        with self.assertRaises((TypeError, ValueError)):
            RfPeTrainWriter(None, self.writer)

    def test_init_raises_when_writer_is_none(self):
        with self.assertRaises((TypeError, ValueError)):
            RfPeTrainWriter(self.mapper, None)

    def test_init_raises_when_mapper_has_wrong_type(self):
        with self.assertRaises((TypeError, ValueError)):
            RfPeTrainWriter("bad-mapper", self.writer)

    def test_init_raises_when_writer_has_wrong_type(self):
        with self.assertRaises((TypeError, ValueError)):
            RfPeTrainWriter(self.mapper, "bad-writer")

    def test_write_output_calls_mapper_and_writer_with_expected_args(self):
        args = MagicMock(spec=RfPeTrainReportArgs)
        args.out_report_md = "out/report.md"
        args.out_model_joblib = "out/model.joblib"
        args.out_schema_json = "out/schema.json"
        args.out_report_json = "out/report.json"
        args.feature_importance_csv = "out/feature_importance.csv"

        result = MagicMock(spec=RfPeTrainResult)
        rf_model = MagicMock()
        result.rf_model = rf_model

        feature_names = ["f1", "f2", "f3"]
        end_message = "# report text"
        report_obj = {
            "cv_splits": 3,
            "acc_mean": 0.8,
            "auc_mean": 0.9,
        }

        self.mapper.get_feature_columns_list = MagicMock(return_value=feature_names)
        self.mapper.get_end_message = MagicMock(return_value=end_message)
        self.mapper.get_report = MagicMock(return_value=report_obj)

        self.writer.write_text_to_text_file = MagicMock()
        self.writer.write_model_to_joblib_file = MagicMock()
        self.writer.write_list_to_json_file = MagicMock()
        self.writer.write_object_to_json_file = MagicMock()
        self.writer.write_feature_importance_to_csv = MagicMock()

        self.train_writer.write_output(args, result)

        self.mapper.get_feature_columns_list.assert_called_once_with(result)
        self.mapper.get_end_message.assert_called_once_with(result)
        self.mapper.get_report.assert_called_once_with(args, result)

        self.writer.write_text_to_text_file.assert_called_once_with(
            args.out_report_md,
            end_message
        )
        self.writer.write_model_to_joblib_file.assert_called_once_with(
            args.out_model_joblib,
            rf_model
        )
        self.writer.write_list_to_json_file.assert_called_once_with(
            args.out_schema_json,
            feature_names
        )
        self.writer.write_object_to_json_file.assert_called_once_with(
            args.out_report_json,
            report_obj
        )
        self.writer.write_feature_importance_to_csv.assert_called_once_with(
            args.feature_importance_csv,
            feature_names,
            rf_model
        )

    def test_write_output_uses_rf_model_from_result(self):
        args = MagicMock(spec=RfPeTrainReportArgs)
        args.out_report_md = "a.md"
        args.out_model_joblib = "a.joblib"
        args.out_schema_json = "a_schema.json"
        args.out_report_json = "a_report.json"
        args.feature_importance_csv = "a.csv"

        result = MagicMock(spec=RfPeTrainResult)
        result.rf_model = MagicMock()

        self.mapper.get_feature_columns_list = MagicMock(return_value=["col1"])
        self.mapper.get_end_message = MagicMock(return_value="msg")
        self.mapper.get_report = MagicMock(return_value={"k": "v"})

        self.writer.write_text_to_text_file = MagicMock()
        self.writer.write_model_to_joblib_file = MagicMock()
        self.writer.write_list_to_json_file = MagicMock()
        self.writer.write_object_to_json_file = MagicMock()
        self.writer.write_feature_importance_to_csv = MagicMock()

        self.train_writer.write_output(args, result)

        self.writer.write_model_to_joblib_file.assert_called_once_with(
            args.out_model_joblib,
            result.rf_model
        )
        self.writer.write_feature_importance_to_csv.assert_called_once_with(
            args.feature_importance_csv,
            ["col1"],
            result.rf_model
        )
