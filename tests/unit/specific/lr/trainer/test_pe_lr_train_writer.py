import unittest
from unittest.mock import MagicMock

from src.specific.lr.trainer.pe_lr_train_writer import LrPeTrainWriter
from src.specific.lr.trainer.pe_lr_train_output_mapper import LrPeTrainOutputMapper
from src.specific.lr.trainer.pe_lr_train_output_writer import LrPeTrainOutputWriter
from src.specific.lr.trainer.pe_lr_train_result import LrPeTrainResult


class TestLrPeTrainWriter(unittest.TestCase):

    def setUp(self):
        self.mapper = object.__new__(LrPeTrainOutputMapper)
        self.writer = object.__new__(LrPeTrainOutputWriter)
        self.train_writer = LrPeTrainWriter(self.mapper, self.writer)

    def test_init_sets_dependencies(self):
        self.assertIs(self.mapper, self.train_writer.output_mapper)
        self.assertIs(self.writer, self.train_writer.output_writer)

    def test_write_calls_mapper_and_writer_with_expected_args(self):
        report = MagicMock()
        output_dir = "/tmp/output"
        train_result = MagicMock(spec=LrPeTrainResult)

        mapped_output = {
            'model': MagicMock(),
            'metrics': {"acc": 0.9},
            'feature_schema': ["f1", "f2"]
        }

        self.mapper.map_report_to_output = MagicMock(return_value=mapped_output)
        self.writer.write_model = MagicMock(return_value="model_path.joblib")
        self.writer.write_metrics = MagicMock(return_value="metrics_path.json")
        self.writer.write_feature_schema = MagicMock(return_value="schema_path.json")

        result = self.train_writer.write(report, output_dir, train_result)

        self.mapper.map_report_to_output.assert_called_once_with(report, train_result)
        self.writer.write_model.assert_called_once_with(output_dir, mapped_output['model'])
        self.writer.write_metrics.assert_called_once_with(output_dir, mapped_output['metrics'])
        self.writer.write_feature_schema.assert_called_once_with(output_dir, mapped_output['feature_schema'])

        self.assertEqual(result['model_path'], "model_path.joblib")
        self.assertEqual(result['metrics_path'], "metrics_path.json")
        self.assertEqual(result['schema_path'], "schema_path.json")

    def test_write_returns_dict_with_all_paths(self):
        report = MagicMock()
        output_dir = "/tmp/output"
        train_result = MagicMock(spec=LrPeTrainResult)

        mapped_output = {
            'model': MagicMock(),
            'metrics': {},
            'feature_schema': []
        }

        self.mapper.map_report_to_output = MagicMock(return_value=mapped_output)
        self.writer.write_model = MagicMock(return_value="model.joblib")
        self.writer.write_metrics = MagicMock(return_value="metrics.json")
        self.writer.write_feature_schema = MagicMock(return_value="schema.json")

        result = self.train_writer.write(report, output_dir, train_result)

        self.assertIsInstance(result, dict)
        self.assertIn('model_path', result)
        self.assertIn('metrics_path', result)
        self.assertIn('schema_path', result)
