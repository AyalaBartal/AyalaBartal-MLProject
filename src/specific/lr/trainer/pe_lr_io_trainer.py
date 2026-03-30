from src.specific.lr.trainer.pe_lr_data_trainer import LrPeDataTrainer
from src.specific.lr.trainer.pe_lr_model_trainer import LrPeModelTrainer
from src.specific.lr.trainer.pe_lr_report_trainer import LrPeReportTrainer
from src.common.preprocessor.pe_dt_preprocessor_provider import DtPePreprocessorProvider


class LrPeIoTrainer:

    def __init__(self, data_reader: LrPeDataTrainer, model_trainer: LrPeModelTrainer, reporter: LrPeReportTrainer):
        self.data_reader = data_reader
        self.model_trainer = model_trainer
        self.reporter = reporter

    def get_ml_features_and_labels(self, data, label_col):
        labels = self.data_reader.get_label_as_series(data, label_col)
        features = self.data_reader.get_features_data_frame(data, label_col)
        return self._preprocess_features(features), labels

    def _preprocess_features(self, features):
        mapper = DtPePreprocessorProvider.get_mapper()
        return mapper.map(features)
