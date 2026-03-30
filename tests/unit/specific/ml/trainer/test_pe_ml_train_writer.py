import unittest
from unittest.mock import MagicMock

from src.specific.ml.trainer.pe_ml_train_writer import MlPeTrainWriter
from src.specific.ml.trainer.pe_ml_train_output_mapper import MlPeTrainOutputMapper
from src.specific.ml.trainer.pe_ml_train_output_writer import MlPeTrainOutputWriter


class TestMlPeTrainWriter(unittest.TestCase):

    def setUp(self):
        self.mapper = MagicMock(spec=MlPeTrainOutputMapper)
        self.writer = MagicMock(spec=MlPeTrainOutputWriter)

        self.train_writer = MlPeTrainWriter(
            output_mapper=self.mapper,
            output_writer=self.writer
        )

    def test_init_stores_dependencies(self):
        self.assertIs(self.mapper, self.train_writer.output_mapper)
        self.assertIs(self.writer, self.train_writer.output_writer)

    def test_write_orchestrates_mapping_and_writing(self):
        report = {
            'cv_auc_mean': 0.85,
            'cv_auc_std': 0.02,
            'cv_accuracy_mean': 0.80,
            'cv_accuracy_std': 0.03,
            'confusion_matrix': [],
            'model': None,
            'feature_names': ['f1', 'f2']
        }
        output_dir = "/tmp/output"
        train_result = object()

        mapped_output = {
            'model': None,
            'metrics': {'cv_auc_mean': 0.85},
            'feature_schema': {'n_features': 2}
        }

        self.mapper.map_report_to_output.return_value = mapped_output
        self.writer.write_model.return_value = "/tmp/output/mlp_model.pt"
        self.writer.write_metrics.return_value = "/tmp/output/ml_cv_metrics.json"
        self.writer.write_feature_schema.return_value = "/tmp/output/ml_feature_schema.json"

        self.train_writer.write(report, output_dir, train_result)

        self.mapper.map_report_to_output.assert_called_once_with(report, train_result)
        self.writer.write_model.assert_called_once()
        self.writer.write_metrics.assert_called_once()
        self.writer.write_feature_schema.assert_called_once()

    def test_write_returns_paths_dict(self):
        report = {
            'cv_auc_mean': 0.85,
            'model': None,
            'feature_names': ['f1']
        }

        mapped_output = {
            'model': None,
            'metrics': {},
            'feature_schema': {}
        }

        self.mapper.map_report_to_output.return_value = mapped_output
        self.writer.write_model.return_value = "model_path"
        self.writer.write_metrics.return_value = "metrics_path"
        self.writer.write_feature_schema.return_value = "schema_path"

        result = self.train_writer.write(report, "/output", object())

        self.assertIn('model_path', result)
        self.assertIn('metrics_path', result)
        self.assertIn('schema_path', result)

    def test_write_passes_correct_arguments_to_writers(self):
        report = {'model': None, 'feature_names': ['f1']}
        mapped_output = {
            'model': 'model_obj',
            'metrics': {'auc': 0.9},
            'feature_schema': {'n_features': 1}
        }

        self.mapper.map_report_to_output.return_value = mapped_output
        self.writer.write_model.return_value = "mp"
        self.writer.write_metrics.return_value = "mep"
        self.writer.write_feature_schema.return_value = "sp"

        self.train_writer.write(report, "/out", object())

        self.writer.write_model.assert_called_once_with("/out", 'model_obj')
        self.writer.write_metrics.assert_called_once_with("/out", {'auc': 0.9})
        self.writer.write_feature_schema.assert_called_once_with("/out", {'n_features': 1})
