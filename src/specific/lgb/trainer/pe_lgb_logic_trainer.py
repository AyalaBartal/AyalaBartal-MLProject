#!/usr/bin/env python3
from pandas import DataFrame

from src.common.validator.args_validator import ArgsValidator
from src.common.preprocessor import DtPePreprocessorProvider
from src.specific.lgb.trainer.pe_lgb_report_trainer import LgbPeReportTrainer
from src.specific.lgb.trainer.pe_lgb_data_trainer import LgbPeDataTrainer
from src.specific.lgb.trainer.pe_lgb_model_trainer import LgbPeModelTrainer
from src.specific.lgb.trainer.pe_lgb_train_algo_args import LgbPeTrainAlgoArgs


class LgbPeLogicTrainer:
    """Training logic and cross-validation for LightGBM."""

    def __init__(self, reader: LgbPeDataTrainer, trainer: LgbPeModelTrainer, reporter: LgbPeReportTrainer):
        ArgsValidator.require_type_not_none(reader, LgbPeDataTrainer, "reader")
        ArgsValidator.require_type_not_none(trainer, LgbPeModelTrainer, "trainer")
        ArgsValidator.require_type_not_none(reporter, LgbPeReportTrainer, "reporter")
        self.data_reader = reader
        self.trainer = trainer
        self.reporter = reporter

    def train(self, args: LgbPeTrainAlgoArgs, data: DataFrame):
        """
        Train LightGBM model following typical ML workflow:
        1. Preprocess raw PE features using DtPePreprocessorProvider
        2. Evaluate model with cross-validation: accuracy and AUC
        3. Train final model on full dataset
        4. Build confusion matrix and generate report
        """
        # Apply preprocessing pipeline to transform raw features
        preprocessor = DtPePreprocessorProvider.get_mapper()
        data = preprocessor.map(data)
        
        ml_label = self.data_reader.get_label_as_series(data, args.label)
        ml_features = self.data_reader.get_features_data_frame(data, args.label)

        model = self.trainer.get_lightgbm_classifier(args)
        skf = self.trainer.get_split_train_test(args)
        scores = self.trainer.get_cross_validate_score(model, skf, ml_features, ml_label)
        model = self.trainer.fit_model(model, ml_features, ml_label)

        con_matrix = self.reporter.get_confusion_matrix(model, skf, ml_features, ml_label)
        return self.reporter.get_report(args, ml_features, model, con_matrix, scores)
