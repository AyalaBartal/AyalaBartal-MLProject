class DtPeDataTrainer:

    def get_features_data_frame(self, data, label_name):
        return data.drop(columns=[label_name])

    def get_label_as_series(self, data, label_name):
        return data[label_name]
