import pandas as pd


class LrPeDataTrainer:

    def read_csv_to_df(self, report_args):
        return pd.read_csv(report_args.input_csv)

    def get_features_data_frame(self, data, label_name):
        return data.drop(columns=[label_name])

    def get_label_as_series(self, data, label_name):
        return data[label_name]

    # Select train/test rows from pandas DataFrame or Series using iloc.
    def select_train_test(self, data, train_idx, test_idx):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        return train_data, test_data
