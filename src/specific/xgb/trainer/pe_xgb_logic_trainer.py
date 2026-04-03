#!/usr/bin/env python3
from pandas import DataFrame

from src.common.validator.args_validator import ArgsValidator
from src.common.preprocessor import DtPePreprocessorProvider
from src.specific.xgb.trainer.pe_xgb_report_trainer import XgbPeReportTrainer
from src.specific.xgb.trainer.pe_xgb_data_trainer import XgbPeDataTrainer
from src.specific.xgb.trainer.pe_xgb_model_trainer import XgbPeModelTrainer
from src.specific.xgb.trainer.pe_xgb_train_algo_args import XgbPeTrainAlgoArgs


class XgbPeLogicTrainer:
    """Training logic and cross-validation for XGBoost."""

    def __init__(self, reader: XgbPeDataTrainer, trainer: XgbPeModelTrainer, reporter: XgbPeReportTrainer):
        ArgsValidator.require_type_not_none(reader, XgbPeDataTrainer, "reader")
        ArgsValidator.require_type_not_none(trainer, XgbPeModelTrainer, "trainer")
        ArgsValidator.require_type_not_none(reporter, XgbPeReportTrainer, "reporter")
        self.data_reader = reader
        self.trainer = trainer
        self.reporter = reporter

    def train(self, args: XgbPeTrainAlgoArgs, data: DataFrame):
        """
        Train XGBoost model following typical ML workflow:
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

        model = self.trainer.get_xgboost_classifier(args)
        skf = self.trainer.get_split_train_test(args)
        scores = self.trainer.get_cross_validate_score(model, skf, ml_features, ml_label)
        model = self.trainer.fit_model(model, ml_features, ml_label)

        con_matrix = self.reporter.get_confusion_matrix(model, skf, ml_features, ml_label)
        return self.reporter.get_report(args, ml_features, model, con_matrix, scores)
