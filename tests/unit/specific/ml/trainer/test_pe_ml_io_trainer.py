import unittest
from unittest.mock import MagicMock, patch
import pandas as pd

from src.specific.ml.trainer.pe_ml_io_trainer import MlPeIoTrainer
from src.specific.ml.trainer.pe_ml_data_trainer import MlPeDataTrainer
from src.specific.ml.trainer.pe_ml_model_trainer import MlPeModelTrainer
from src.specific.ml.trainer.pe_ml_report_trainer import MlPeReportTrainer


class TestMlPeIoTrainer(unittest.TestCase):

    def setUp(self):
        self.data_reader = MagicMock(spec=MlPeDataTrainer)
        self.model_trainer = MagicMock(spec=MlPeModelTrainer)
        self.reporter = MagicMock(spec=MlPeReportTrainer)

        self.io_trainer = MlPeIoTrainer(
            data_reader=self.data_reader,
            model_trainer=self.model_trainer,
            reporter=self.reporter
        )

    def test_init_stores_dependencies(self):
        self.assertIs(self.data_reader, self.io_trainer.data_reader)
        self.assertIs(self.model_trainer, self.io_trainer.model_trainer)
        self.assertIs(self.reporter, self.io_trainer.reporter)

    @patch('src.specific.ml.trainer.pe_ml_io_trainer.DtPePreprocessorProvider')
    def test_get_ml_features_and_labels_calls_data_reader(self, mock_provider):
        data = pd.DataFrame({
            "f1": [1, 2, 3],
            "f2": [4, 5, 6],
            "label": [0, 1, 0]
        })

        expected_labels = pd.Series([0, 1, 0])
        expected_features = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
        preprocessed_features = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})

        self.data_reader.get_label_as_series.return_value = expected_labels
        self.data_reader.get_features_data_frame.return_value = expected_features

        mock_mapper = MagicMock()
        mock_mapper.map.return_value = preprocessed_features
        mock_provider.get_mapper.return_value = mock_mapper

        features, labels = self.io_trainer.get_ml_features_and_labels(data, "label")

        self.data_reader.get_label_as_series.assert_called_once_with(data, "label")
        self.data_reader.get_features_data_frame.assert_called_once_with(data, "label")

    @patch('src.specific.ml.trainer.pe_ml_io_trainer.DtPePreprocessorProvider')
    def test_get_ml_features_and_labels_returns_tuple(self, mock_provider):
        data = pd.DataFrame({"f1": [1], "label": [0]})

        expected_labels = pd.Series([0])
        expected_features = pd.DataFrame({"f1": [1]})
        preprocessed_features = pd.DataFrame({"f1": [1]})

        self.data_reader.get_label_as_series.return_value = expected_labels
        self.data_reader.get_features_data_frame.return_value = expected_features

        mock_mapper = MagicMock()
        mock_mapper.map.return_value = preprocessed_features
        mock_provider.get_mapper.return_value = mock_mapper

        result = self.io_trainer.get_ml_features_and_labels(data, "label")

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    @patch('src.specific.ml.trainer.pe_ml_io_trainer.DtPePreprocessorProvider')
    def test_get_ml_features_and_labels_preserves_label_values(self, mock_provider):
        data = pd.DataFrame({"f1": [1, 2], "label": [1, 0]})

        expected_labels = pd.Series([1, 0])
        expected_features = pd.DataFrame({"f1": [1, 2]})
        preprocessed_features = pd.DataFrame({"f1": [1, 2]})

        self.data_reader.get_label_as_series.return_value = expected_labels
        self.data_reader.get_features_data_frame.return_value = expected_features

        mock_mapper = MagicMock()
        mock_mapper.map.return_value = preprocessed_features
        mock_provider.get_mapper.return_value = mock_mapper

        features, labels = self.io_trainer.get_ml_features_and_labels(data, "label")

        pd.testing.assert_series_equal(labels, expected_labels)
