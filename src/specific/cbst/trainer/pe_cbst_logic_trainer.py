#!/usr/bin/env python3
from pandas import DataFrame

from src.common.validator.args_validator import ArgsValidator
from src.common.preprocessor import DtPePreprocessorProvider
from src.specific.cbst.trainer.pe_cbst_report_trainer import CbstPeReportTrainer
from src.specific.cbst.trainer.pe_cbst_data_trainer import CbstPeDataTrainer
from src.specific.cbst.trainer.pe_cbst_model_trainer import CbstPeModelTrainer
from src.specific.cbst.trainer.pe_cbst_train_algo_args import CbstPeTrainAlgoArgs


class CbstPeLogicTrainer:
    """Training logic and cross-validation for CatBoost."""

    def __init__(self, reader: CbstPeDataTrainer, trainer: CbstPeModelTrainer, reporter: CbstPeReportTrainer):
        ArgsValidator.require_type_not_none(reader, CbstPeDataTrainer, "reader")
        ArgsValidator.require_type_not_none(trainer, CbstPeModelTrainer, "trainer")
        ArgsValidator.require_type_not_none(reporter, CbstPeReportTrainer, "reporter")
        self.data_reader = reader
        self.trainer = trainer
        self.reporter = reporter

    def train(self, args: CbstPeTrainAlgoArgs, data: DataFrame):
        """
        Train CatBoost model following typical ML workflow:
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

        model = self.trainer.get_catboost_classifier(args)
        skf = self.trainer.get_split_train_test(args)
        scores = self.trainer.get_cross_validate_score(model, skf, ml_features, ml_label)
        model = self.trainer.fit_model(model, ml_features, ml_label)

        con_matrix = self.reporter.get_confusion_matrix(model, skf, ml_features, ml_label)
        return self.reporter.get_report(args, ml_features, model, con_matrix, scores)
