import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.specific.dt.trainer.pe_dt_train_output_writer import DtPeTrainOutputWriter


class TestDtPeTrainOutputWriter(unittest.TestCase):

    def setUp(self):
        self.writer = DtPeTrainOutputWriter()

    def test_write_text_to_text_file_writes_expected_content(self):
        expected = "# hello\nsome text"

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "report.md"

            self.writer.write_text_to_text_file(out_file, expected)

            actual = out_file.read_text(encoding="utf-8")
            self.assertEqual(expected, actual)

    def test_write_model_to_joblib_file_calls_joblib_dump(self):
        import src.specific.dt.trainer.pe_dt_train_output_writer as writer_module

        out_file = "model.joblib"
        dt_model = MagicMock()

        # Create fake joblib with a mock dump
        mock_dump = MagicMock()
        original_joblib = writer_module.joblib

        try:
            writer_module.joblib = MagicMock(dump=mock_dump)

            self.writer.write_model_to_joblib_file(out_file, dt_model)

            mock_dump.assert_called_once_with(dt_model, out_file)

        finally:
            # Restore original joblib to avoid side effects
            writer_module.joblib = original_joblib

    def test_write_object_to_json_file_writes_expected_json(self):
        data = {
            "cv_splits": 3,
            "acc_mean": 0.85,
            "auc_mean": 0.91,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "metrics.json"

            self.writer.write_object_to_json_file(out_file, data)

            with open(out_file, "r", encoding="utf-8") as f:
                actual = json.load(f)

            self.assertEqual(data, actual)

    def test_write_list_to_json_file_writes_feature_order_object(self):
        data = ["f1", "f2", "f3"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "feature_order.json"

            self.writer.write_list_to_json_file(out_file, data)

            with open(out_file, "r", encoding="utf-8") as f:
                actual = json.load(f)

            self.assertEqual({"feature_order": ["f1", "f2", "f3"]}, actual)

    def test_write_list_to_json_file_converts_input_to_list(self):
        data = ("a", "b", "c")

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "feature_order.json"

            self.writer.write_list_to_json_file(out_file, data)

            with open(out_file, "r", encoding="utf-8") as f:
                actual = json.load(f)

            self.assertEqual({"feature_order": ["a", "b", "c"]}, actual)

    def test_write_dt_model_to_graphviz_dot_file_calls_export_graphviz_with_expected_args(self):
        import src.specific.dt.trainer.pe_dt_train_output_writer as writer_module

        out_file = Path("tree.dot")
        feature_names = ["f1", "f2", "f3"]

        dt_model = MagicMock()
        dt_model.classes_ = [0, 1]

        mock_export = MagicMock()
        original_export = writer_module.export_graphviz

        try:
            # Replace export_graphviz in the module
            writer_module.export_graphviz = mock_export

            self.writer.write_dt_model_to_graphviz_dot_file(out_file, feature_names, dt_model)

            mock_export.assert_called_once_with(
                dt_model,
                out_file=str(out_file),
                feature_names=feature_names,
                class_names=["0", "1"],
                filled=True,
                rounded=True
            )
        finally:
            # Restore original function
            writer_module.export_graphviz = original_export

    def test_write_feature_importance_to_csv_writes_expected_csv_sorted_desc(self):
        feature_names = ["f1", "f2", "f3"]

        dt_model = MagicMock()
        dt_model.feature_importances_ = [0.2, 0.7, 0.1]

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "feature_importance.csv"

            self.writer.write_feature_importance_to_csv(out_file, feature_names, dt_model)

            actual = pd.read_csv(out_file)

            expected = pd.DataFrame({
                "feature": ["f2", "f1", "f3"],
                "importance": [0.7, 0.2, 0.1]
            })

            pd.testing.assert_frame_equal(expected, actual)

    def test_write_feature_importance_to_csv_writes_without_index_column(self):
        feature_names = ["a", "b"]

        dt_model = MagicMock()
        dt_model.feature_importances_ = [0.4, 0.6]

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "feature_importance.csv"

            self.writer.write_feature_importance_to_csv(out_file, feature_names, dt_model)

            actual = pd.read_csv(out_file)

            self.assertEqual(["feature", "importance"], list(actual.columns))
            self.assertNotIn("Unnamed: 0", actual.columns)
