import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.specific.lr.trainer.pe_lr_train_output_writer import LrPeTrainOutputWriter


class TestLrPeTrainOutputWriter(unittest.TestCase):

    def setUp(self):
        self.writer = LrPeTrainOutputWriter()

    def test_write_model_writes_model_to_joblib_file(self):
        import src.specific.lr.trainer.pe_lr_train_output_writer as writer_module

        with tempfile.TemporaryDirectory() as tmp_dir:
            lr_model = MagicMock()
            
            mock_dump = MagicMock()
            original_dump = writer_module.dump

            try:
                writer_module.dump = mock_dump

                self.writer.write_model(tmp_dir, lr_model)

                mock_dump.assert_called_once()
                args, kwargs = mock_dump.call_args
                self.assertIs(lr_model, args[0])
                self.assertIn("logistic_regression_model.joblib", args[1])

            finally:
                writer_module.dump = original_dump

    def test_write_metrics_writes_expected_json(self):
        data = {
            "cv_splits": 3,
            "acc_mean": 0.85,
            "auc_mean": 0.91,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = self.writer.write_metrics(tmp_dir, data)

            self.assertTrue(result_path.endswith("lr_cv_metrics.json"))

            with open(result_path, "r", encoding="utf-8") as f:
                actual = json.load(f)

            self.assertEqual(data, actual)

    def test_write_feature_schema_writes_expected_json(self):
        schema = ["f1", "f2", "f3"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = self.writer.write_feature_schema(tmp_dir, schema)

            self.assertTrue(result_path.endswith("lr_feature_schema.json"))

            with open(result_path, "r", encoding="utf-8") as f:
                actual = json.load(f)

            self.assertEqual(schema, actual)

    def test_write_model_creates_directory_if_not_exists(self):
        import src.specific.lr.trainer.pe_lr_train_output_writer as writer_module

        with tempfile.TemporaryDirectory() as tmp_dir:
            nested_dir = Path(tmp_dir) / "nested" / "output"
            lr_model = MagicMock()

            mock_dump = MagicMock()
            original_dump = writer_module.dump

            try:
                writer_module.dump = mock_dump

                self.writer.write_model(str(nested_dir), lr_model)

                self.assertTrue(nested_dir.exists())

            finally:
                writer_module.dump = original_dump

    def test_write_metrics_creates_directory_if_not_exists(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            nested_dir = Path(tmp_dir) / "nested" / "output"
            data = {"acc": 0.9}

            result_path = self.writer.write_metrics(str(nested_dir), data)

            self.assertTrue(nested_dir.exists())
            self.assertTrue(Path(result_path).exists())

    def test_write_feature_schema_creates_directory_if_not_exists(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            nested_dir = Path(tmp_dir) / "nested" / "output"
            schema = ["feature1"]

            result_path = self.writer.write_feature_schema(str(nested_dir), schema)

            self.assertTrue(nested_dir.exists())
            self.assertTrue(Path(result_path).exists())
