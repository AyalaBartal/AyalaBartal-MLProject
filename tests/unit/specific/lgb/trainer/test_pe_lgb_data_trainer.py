import unittest
import pandas as pd
import tempfile
import os

from src.specific.lgb.trainer.pe_lgb_data_trainer import LgbPeDataTrainer


class DummyArgs:
    def __init__(self, input_csv):
        self.input_csv = input_csv


class TestLgbPeDataTrainer(unittest.TestCase):

    def setUp(self):
        self.trainer = LgbPeDataTrainer()

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

    def test_read_csv_to_df_reads_from_correct_path(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp:
            self.df.to_csv(tmp.name, index=False)
            args = DummyArgs(tmp.name)

        try:
            result = self.trainer.read_csv_to_df(args)
            self.assertEqual(len(result), 4)
            self.assertEqual(list(result.columns), ["feature1", "feature2", "label"])

        finally:
            os.remove(tmp.name)

    # ---------- get_features_data_frame ----------

    def test_get_features_data_frame_drops_label_column(self):
        result = self.trainer.get_features_data_frame(self.df, "label")

        self.assertNotIn("label", result.columns)
        self.assertListEqual(["feature1", "feature2"], list(result.columns))

    def test_get_features_data_frame_preserves_data_types(self):
        df_typed = pd.DataFrame({
            "f1": [1, 2, 3],
            "f2": [1.5, 2.5, 3.5],
            "label": [0, 1, 0]
        })
        
        result = self.trainer.get_features_data_frame(df_typed, "label")
        
        self.assertEqual(result["f1"].dtype, df_typed["f1"].dtype)
        self.assertEqual(result["f2"].dtype, df_typed["f2"].dtype)

    def test_get_features_data_frame_with_multiple_columns(self):
        df = pd.DataFrame({
            "a": [1, 2],
            "b": [3, 4],
            "c": [5, 6],
            "label": [0, 1]
        })
        
        result = self.trainer.get_features_data_frame(df, "label")
        
        self.assertEqual(len(result.columns), 3)
        self.assertListEqual(list(result.columns), ["a", "b", "c"])

    # ---------- get_label_as_series ----------

    def test_get_label_as_series_returns_correct_series(self):
        result = self.trainer.get_label_as_series(self.df, "label")

        self.assertIsInstance(result, pd.Series)
        pd.testing.assert_series_equal(self.df["label"], result)

    def test_get_label_as_series_preserves_values(self):
        result = self.trainer.get_label_as_series(self.df, "label")
        
        expected_values = [0, 1, 0, 1]
        self.assertEqual(list(result.values), expected_values)

    def test_get_label_as_series_with_different_label_column(self):
        df = pd.DataFrame({
            "feature": [1, 2, 3],
            "target": [0, 1, 0]
        })
        
        result = self.trainer.get_label_as_series(df, "target")
        
        pd.testing.assert_series_equal(df["target"], result)

    # ---------- select_train_test (DataFrame) ----------

    def test_select_train_test_with_dataframe(self):
        train_idx = [0, 1]
        test_idx = [2, 3]

        train_df, test_df = self.trainer.select_train_test(self.df, train_idx, test_idx)

        pd.testing.assert_frame_equal(train_df, self.df.iloc[train_idx])
        pd.testing.assert_frame_equal(test_df, self.df.iloc[test_idx])

    def test_select_train_test_dataframe_correct_shapes(self):
        train_idx = [0, 2]
        test_idx = [1, 3]

        train_df, test_df = self.trainer.select_train_test(self.df, train_idx, test_idx)

        self.assertEqual(len(train_df), 2)
        self.assertEqual(len(test_df), 2)

    def test_select_train_test_dataframe_disjoint_indices(self):
        train_idx = [0]
        test_idx = [1, 2, 3]

        train_df, test_df = self.trainer.select_train_test(self.df, train_idx, test_idx)

        self.assertEqual(len(train_df), 1)
        self.assertEqual(len(test_df), 3)

    # ---------- select_train_test (Series) ----------

    def test_select_train_test_with_series(self):
        series = self.df["label"]

        train_idx = [0, 2]
        test_idx = [1, 3]

        train_s, test_s = self.trainer.select_train_test(series, train_idx, test_idx)

        pd.testing.assert_series_equal(train_s, series.iloc[train_idx])
        pd.testing.assert_series_equal(test_s, series.iloc[test_idx])

    def test_select_train_test_series_correct_values(self):
        series = pd.Series([0, 1, 0, 1, 0])
        train_idx = [0, 2, 4]
        test_idx = [1, 3]

        train_s, test_s = self.trainer.select_train_test(series, train_idx, test_idx)

        self.assertEqual(list(train_s.values), [0, 0, 0])
        self.assertEqual(list(test_s.values), [1, 1])

    def test_select_train_test_series_preserves_index(self):
        series = self.df["label"]
        train_idx = [0, 1]
        test_idx = [2, 3]

        train_s, test_s = self.trainer.select_train_test(series, train_idx, test_idx)

        self.assertEqual(list(train_s.index), [0, 1])
        self.assertEqual(list(test_s.index), [2, 3])

    def test_select_train_test_empty_indices(self):
        train_idx = []
        test_idx = [0, 1, 2, 3]

        train_df, test_df = self.trainer.select_train_test(self.df, train_idx, test_idx)

        self.assertEqual(len(train_df), 0)
        self.assertEqual(len(test_df), 4)


if __name__ == "__main__":
    unittest.main()
