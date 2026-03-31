import unittest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import MagicMock
import torch
import torch.nn as nn

from src.specific.ml.trainer.pe_ml_train_output_writer import MlPeTrainOutputWriter


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


class TestMlPeTrainOutputWriter(unittest.TestCase):

    def setUp(self):
        self.writer = MlPeTrainOutputWriter()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_write_model_creates_directory(self):
        model = DummyModel()

        self.writer.write_model(self.temp_dir, model)

        self.assertTrue(os.path.exists(self.temp_dir))

    def test_write_model_returns_model_path(self):
        model = DummyModel()

        path = self.writer.write_model(self.temp_dir, model)

        self.assertIsInstance(path, str)
        self.assertIn('mlp_model.pt', path)

    def test_write_model_saves_torch_file(self):
        model = DummyModel()

        path = self.writer.write_model(self.temp_dir, model)

        self.assertTrue(os.path.exists(path))

    def test_write_metrics_creates_json_file(self):
        metrics = {
            'cv_auc_mean': 0.85,
            'cv_auc_std': 0.02,
            'cv_accuracy_mean': 0.80,
            'cv_accuracy_std': 0.03
        }

        path = self.writer.write_metrics(self.temp_dir, metrics)

        self.assertTrue(os.path.exists(path))
        self.assertTrue(path.endswith('ml_cv_metrics.json'))

    def test_write_metrics_writes_correct_content(self):
        metrics = {
            'cv_auc_mean': 0.85,
            'cv_auc_std': 0.02
        }

        path = self.writer.write_metrics(self.temp_dir, metrics)

        with open(path, 'r') as f:
            loaded = json.load(f)

        self.assertEqual(loaded['cv_auc_mean'], 0.85)
        self.assertEqual(loaded['cv_auc_std'], 0.02)

    def test_write_metrics_returns_metrics_path(self):
        metrics = {'cv_auc_mean': 0.85}

        path = self.writer.write_metrics(self.temp_dir, metrics)

        self.assertIsInstance(path, str)
        self.assertIn('ml_cv_metrics.json', path)

    def test_write_feature_schema_creates_json_file(self):
        schema = {
            'feature_order': ['f1', 'f2', 'f3'],
            'n_features': 3
        }

        path = self.writer.write_feature_schema(self.temp_dir, schema)

        self.assertTrue(os.path.exists(path))
        self.assertTrue(path.endswith('ml_feature_schema.json'))

    def test_write_feature_schema_writes_correct_content(self):
        schema = {
            'feature_order': ['feat_a', 'feat_b'],
            'n_features': 2
        }

        path = self.writer.write_feature_schema(self.temp_dir, schema)

        with open(path, 'r') as f:
            loaded = json.load(f)

        self.assertEqual(loaded['feature_order'], ['feat_a', 'feat_b'])
        self.assertEqual(loaded['n_features'], 2)

    def test_write_feature_schema_returns_schema_path(self):
        schema = {'feature_order': ['f1'], 'n_features': 1}

        path = self.writer.write_feature_schema(self.temp_dir, schema)

        self.assertIsInstance(path, str)
        self.assertIn('ml_feature_schema.json', path)
