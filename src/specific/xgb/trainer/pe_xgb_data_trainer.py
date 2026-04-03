import pandas as pd


class XgbPeDataTrainer:
    """Data loading and preprocessing for XGBoost trainer."""

    def read_csv_to_df(self, report_args):
        """Read CSV file into pandas DataFrame."""
        return pd.read_csv(report_args.input_csv)

    def get_features_data_frame(self, data, label_name):
        """Extract feature columns, excluding label."""
        return data.drop(columns=[label_name])

    def get_label_as_series(self, data, label_name):
        """Extract label column as pandas Series."""
        return data[label_name]

    def select_train_test(self, data, train_idx, test_idx):
        """Select train/test rows from pandas DataFrame or Series using iloc."""
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        return train_data, test_data
