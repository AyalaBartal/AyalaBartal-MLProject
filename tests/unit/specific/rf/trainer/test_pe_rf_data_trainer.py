import unittest
import pandas as pd
import tempfile
import os

from src.specific.rf.trainer.pe_rf_data_trainer import RfPeDataTrainer


class DummyArgs:
    def __init__(self, input_csv):
        self.input_csv = input_csv


class TestRfPeDataTrainer(unittest.TestCase):

    def setUp(self):
        self.trainer = RfPeDataTrainer()

        # Sample DataFrame
        self.df = pd.DataFrame({
            "feature1": [1, 2, 3, 4],
            "feature2": [10, 20, 30, 40],
            "label": [0, 1, 0, 1]
        })

    # ---------- read_csv_to_df ----------

    def test_read_csv_to_df_returns_dataframe(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp:
            self.df.to_csv(tmp.name, index=False)
            args = DummyArgs(tmp.name)

        try:
            result = self.trainer.read_csv_to_df(args)

            self.assertIsInstance(result, pd.DataFrame)
            pd.testing.assert_frame_equal(self.df, result)

        finally:
            os.remove(tmp.name)

    # ---------- get_features_data_frame ----------

    def test_get_features_data_frame_drops_label_column(self):
        result = self.trainer.get_features_data_frame(self.df, "label")

        self.assertNotIn("label", result.columns)
        self.assertListEqual(["feature1", "feature2"], list(result.columns))

    # ---------- get_label_as_series ----------

    def test_get_label_as_series_returns_correct_series(self):
        result = self.trainer.get_label_as_series(self.df, "label")

        self.assertIsInstance(result, pd.Series)
        pd.testing.assert_series_equal(self.df["label"], result)

    # ---------- select_train_test (DataFrame) ----------

    def test_select_train_test_with_dataframe(self):
        train_idx = [0, 1]
        test_idx = [2, 3]

        train_df, test_df = self.trainer.select_train_test(self.df, train_idx, test_idx)

        pd.testing.assert_frame_equal(train_df, self.df.iloc[train_idx])
        pd.testing.assert_frame_equal(test_df, self.df.iloc[test_idx])

    # ---------- select_train_test (Series) ----------

    def test_select_train_test_with_series(self):
        series = self.df["label"]

        train_idx = [0, 2]
        test_idx = [1, 3]

        train_s, test_s = self.trainer.select_train_test(series, train_idx, test_idx)

        pd.testing.assert_series_equal(train_s, series.iloc[train_idx])
        pd.testing.assert_series_equal(test_s, series.iloc[test_idx])
