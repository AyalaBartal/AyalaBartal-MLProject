import unittest
import numpy as np
import pandas as pd

from src.specific.dt.trainer.pe_dt_train_output_mapper import DtPeTrainOutputMapper
from src.specific.dt.trainer.pe_dt_train_result import DtPeTrainResult
from src.specific.dt.trainer.pe_dt_train_algo_args import DtPeTrainAlgoArgs


class TestDtPeTrainOutputMapper(unittest.TestCase):

    def setUp(self):
        self.mapper = DtPeTrainOutputMapper()

        self.args = DtPeTrainAlgoArgs()
        self.args.n_splits = 3
        self.args.criterion = "gini"
        self.args.max_depth = 24
        self.args.min_samples_leaf = 2

        self.input_features = pd.DataFrame({
            "f1": [1, 2, 3, 4],
            "f2": [10, 20, 30, 40],
            "f3": [100, 200, 300, 400],
        })

        self.confusion_matrix = np.array([
            [5, 1],
            [2, 6]
        ])

        self.acc_score = np.array([0.70, 0.80, 0.90])
        self.auc_score = np.array([0.75, 0.85, 0.95])

        self.dt_model = object()

        self.result = DtPeTrainResult(
            self.args,
            self.input_features,
            self.dt_model,
            self.confusion_matrix,
            self.acc_score,
            self.auc_score,
        )

    def test_get_end_message_returns_expected_text(self):
        actual = self.mapper.get_end_message(self.result)

        expected = "\n".join([
            "# Decision Tree — Cross-Validation",
            "",
            "- Splits: 3",
            f"- AUC (mean ± std): {np.mean(self.auc_score):.4f} ± {np.std(self.auc_score, ddof=1):.4f}",
            f"- Accuracy (mean ± std): {np.mean(self.acc_score):.4f} ± {np.std(self.acc_score, ddof=1):.4f}",
            "- Samples: 4, Features: 3",
            "- Model: criterion=gini, max_depth=24, min_sample_leaf=2",
            ""
        ])

        self.assertEqual(expected, actual)

    def test_get_end_message_contains_key_sections(self):
        actual = self.mapper.get_end_message(self.result)

        self.assertIn("# Decision Tree — Cross-Validation", actual)
        self.assertIn("- Splits: 3", actual)
        self.assertIn("- Samples: 4, Features: 3", actual)
        self.assertIn("- Model: criterion=gini, max_depth=24, min_sample_leaf=2", actual)

    def test_get_report_returns_expected_dict(self):
        actual = self.mapper.get_report(self.args, self.result)

        expected = {
            "cv_splits": 3,
            "cm": [[5, 1], [2, 6]],
            "auc_mean": float(np.mean(self.auc_score)),
            "auc_std": float(np.std(self.auc_score, ddof=1)),
            "acc_mean": float(np.mean(self.acc_score)),
            "acc_std": float(np.std(self.acc_score, ddof=1)),
            "n_features": 3,
            "n_samples": 4,
        }

        self.assertEqual(expected, actual)

    def test_get_report_converts_confusion_matrix_to_list(self):
        actual = self.mapper.get_report(self.args, self.result)

        self.assertIsInstance(actual["cm"], list)
        self.assertEqual([[5, 1], [2, 6]], actual["cm"])

    def test_get_feature_columns_list_returns_columns_in_order(self):
        actual = self.mapper.get_feature_columns_list(self.result)

        self.assertEqual(["f1", "f2", "f3"], actual)

    def test_get_feature_columns_list_returns_empty_list_when_no_columns(self):
        empty_features = pd.DataFrame()
        result = DtPeTrainResult(
            self.args,
            empty_features,
            self.dt_model,
            self.confusion_matrix,
            self.acc_score,
            self.auc_score,
        )

        actual = self.mapper.get_feature_columns_list(result)

        self.assertEqual([], actual)
