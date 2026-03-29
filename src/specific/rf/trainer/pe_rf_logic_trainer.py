#!/usr/bin/env python3
from pandas import DataFrame

from src.common.validator.args_validator import ArgsValidator
from src.common.preprocessor import DtPePreprocessorProvider
from src.specific.rf.trainer.pe_rf_report_trainer import RfPeReportTrainer
from src.specific.rf.trainer.pe_rf_data_trainer import RfPeDataTrainer
from src.specific.rf.trainer.pe_rf_model_trainer import RfPeModelTrainer
from src.specific.rf.trainer.pe_rf_train_algo_args import RfPeTrainAlgoArgs


class RfPeLogicTrainer:

    def __init__(self, reader: RfPeDataTrainer, trainer: RfPeModelTrainer, reporter: RfPeReportTrainer):
        ArgsValidator.require_type_not_none(reader, RfPeDataTrainer, "reader")
        ArgsValidator.require_type_not_none(trainer, RfPeModelTrainer, "trainer")
        ArgsValidator.require_type_not_none(reporter, RfPeReportTrainer, "reporter")
        self.data_reader = reader
        self.trainer = trainer
        self.reporter = reporter

    # Typical ML algorithm model builder workflow:
    # 1 Preprocess raw PE features using DtPePreprocessorProvider
    # 2 Evaluate model with cross-validation methods: acc and auc.
    # 3 Choose best parameters for random forest based on step 1 result.
    # 4 Train final model on full dataset.
    # 5 Run model on full data and build confusion matrix and report.
    def train(self, args: RfPeTrainAlgoArgs, data: DataFrame):
        # Apply preprocessing pipeline to transform raw features
        preprocessor = DtPePreprocessorProvider.get_mapper()
        data = preprocessor.map(data)
        
        ml_label = self.data_reader.get_label_as_series(data, args.label)
        ml_features = self.data_reader.get_features_data_frame(data, args.label)

        model = self.trainer.get_random_forest_classifier(args)
        skf = self.trainer.get_split_train_test(args)
        scores = self.trainer.get_cross_validate_score(model, skf, ml_features, ml_label)
        model = self.trainer.fit_model(model, ml_features, ml_label)

        con_matrix = self.reporter.get_confusion_matrix(model, skf, ml_features, ml_label)
        return self.reporter.get_report(args, ml_features, model, con_matrix, scores)
