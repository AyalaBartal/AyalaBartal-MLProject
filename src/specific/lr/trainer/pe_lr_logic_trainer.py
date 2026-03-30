#!/usr/bin/env python3
from pandas import DataFrame

from src.common.validator.args_validator import ArgsValidator
from src.specific.lr.trainer.pe_lr_report_trainer import LrPeReportTrainer
from src.specific.lr.trainer.pe_lr_data_trainer import LrPeDataTrainer
from src.specific.lr.trainer.pe_lr_model_trainer import LrPeModelTrainer
from src.specific.lr.trainer.pe_lr_train_algo_args import LrPeTrainAlgoArgs


class LrPeLogicTrainer:

    def __init__(self, reader: LrPeDataTrainer, trainer: LrPeModelTrainer, reporter: LrPeReportTrainer):
        ArgsValidator.require_type_not_none(reader, LrPeDataTrainer, "reader")
        ArgsValidator.require_type_not_none(trainer, LrPeModelTrainer, "trainer")
        ArgsValidator.require_type_not_none(reporter, LrPeReportTrainer, "reporter")
        self.data_reader = reader
        self.trainer = trainer
        self.reporter = reporter

    def train(self, args: LrPeTrainAlgoArgs, data: DataFrame):
        ml_label = self.data_reader.get_label_as_series(data, args.label)
        ml_features = self.data_reader.get_features_data_frame(data, args.label)

        model = self.trainer.get_logistic_regression_classifier(args)
        skf = self.trainer.get_split_train_test(args)
        scores = self.trainer.get_cross_validate_score(model, skf, ml_features, ml_label)
        model = self.trainer.fit_model(model, ml_features, ml_label)

        con_matrix = self.reporter.get_confusion_matrix(model, skf, ml_features, ml_label)
        return self.reporter.get_report(args, ml_features, model, con_matrix, scores)
