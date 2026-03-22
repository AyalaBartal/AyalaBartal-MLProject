import unittest
from unittest.mock import MagicMock

from src.specific.dt.trainer.pe_dt_train_writer import DtPeTrainWriter
from src.specific.dt.trainer.pe_dt_train_output_mapper import DtPeTrainOutputMapper
from src.specific.dt.trainer.pe_dt_train_output_writer import DtPeTrainOutputWriter
from src.specific.dt.trainer.pe_dt_train_result import DtPeTrainResult
from src.specific.dt.trainer.pe_dt_train_report_args import DtPeTrainReportArgs


class TestDtPeTrainWriter(unittest.TestCase):

    def setUp(self):
        self.mapper = object.__new__(DtPeTrainOutputMapper)
        self.writer = object.__new__(DtPeTrainOutputWriter)
        self.train_writer = DtPeTrainWriter(self.mapper, self.writer)

    def test_init_sets_dependencies(self):
        self.assertIs(self.mapper, self.train_writer.mapper)
        self.assertIs(self.writer, self.train_writer.writer)

    def test_init_raises_when_mapper_is_none(self):
        with self.assertRaises((TypeError, ValueError)):
            DtPeTrainWriter(None, self.writer)

    def test_init_raises_when_writer_is_none(self):
        with self.assertRaises((TypeError, ValueError)):
            DtPeTrainWriter(self.mapper, None)

    def test_init_raises_when_mapper_has_wrong_type(self):
        with self.assertRaises((TypeError, ValueError)):
            DtPeTrainWriter("bad-mapper", self.writer)

    def test_init_raises_when_writer_has_wrong_type(self):
        with self.assertRaises((TypeError, ValueError)):
            DtPeTrainWriter(self.mapper, "bad-writer")

    def test_write_output_calls_mapper_and_writer_with_expected_args(self):
        args = MagicMock(spec=DtPeTrainReportArgs)
        args.out_report_md = "out/report.md"
        args.out_model_joblib = "out/model.joblib"
        args.out_schema_json = "out/schema.json"
        args.out_report_json = "out/report.json"
        args.model_output_dot = "out/model.dot"
        args.feature_importance_csv = "out/feature_importance.csv"

        result = MagicMock(spec=DtPeTrainResult)
        dt_model = MagicMock()
        result.dt_model = dt_model

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
        self.writer.write_dt_model_to_graphviz_dot_file = MagicMock()
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
            dt_model
        )
        self.writer.write_list_to_json_file.assert_called_once_with(
            args.out_schema_json,
            feature_names
        )
        self.writer.write_object_to_json_file.assert_called_once_with(
            args.out_report_json,
            report_obj
        )
        self.writer.write_dt_model_to_graphviz_dot_file.assert_called_once_with(
            args.model_output_dot,
            feature_names,
            dt_model
        )
        self.writer.write_feature_importance_to_csv.assert_called_once_with(
            args.feature_importance_csv,
            feature_names,
            dt_model
        )

    def test_write_output_uses_dt_model_from_result(self):
        args = MagicMock(spec=DtPeTrainReportArgs)
        args.out_report_md = "a.md"
        args.out_model_joblib = "a.joblib"
        args.out_schema_json = "a_schema.json"
        args.out_report_json = "a_report.json"
        args.model_output_dot = "a.dot"
        args.feature_importance_csv = "a.csv"

        result = MagicMock(spec=DtPeTrainResult)
        result.dt_model = MagicMock()

        self.mapper.get_feature_columns_list = MagicMock(return_value=["col1"])
        self.mapper.get_end_message = MagicMock(return_value="msg")
        self.mapper.get_report = MagicMock(return_value={"k": "v"})

        self.writer.write_text_to_text_file = MagicMock()
        self.writer.write_model_to_joblib_file = MagicMock()
        self.writer.write_list_to_json_file = MagicMock()
        self.writer.write_object_to_json_file = MagicMock()
        self.writer.write_dt_model_to_graphviz_dot_file = MagicMock()
        self.writer.write_feature_importance_to_csv = MagicMock()

        self.train_writer.write_output(args, result)

        self.writer.write_model_to_joblib_file.assert_called_once_with(
            args.out_model_joblib,
            result.dt_model
        )
        self.writer.write_dt_model_to_graphviz_dot_file.assert_called_once_with(
            args.model_output_dot,
            ["col1"],
            result.dt_model
        )
        self.writer.write_feature_importance_to_csv.assert_called_once_with(
            args.feature_importance_csv,
            ["col1"],
            result.dt_model
        )
