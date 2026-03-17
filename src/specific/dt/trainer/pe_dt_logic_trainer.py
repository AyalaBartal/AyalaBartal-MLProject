#!/usr/bin/env python3
from pandas import DataFrame

from src.specific.dt.trainer.args_validator import ArgsValidator
from src.specific.dt.trainer.pe_dt_report_trainer import DtPeReportTrainer
from src.specific.dt.trainer.pe_dt_data_trainer import DtPeDataTrainer
from src.specific.dt.trainer.pe_dt_model_trainer import DtPeModelTrainer
from src.specific.dt.trainer.pe_dt_train_algo_args import DtPeTrainAlgoArgs


class DtPeLogicTrainer:

    def __init__(self, reader: DtPeDataTrainer, trainer: DtPeModelTrainer, reporter: DtPeReportTrainer):
        ArgsValidator.require_type_not_none(reader, DtPeDataTrainer, "reader")
        ArgsValidator.require_type_not_none(trainer, DtPeModelTrainer, "trainer")
        ArgsValidator.require_type_not_none(reporter, DtPeReportTrainer, "reporter")
        self.data_reader = reader
        self.trainer = trainer
        self.reporter = reporter

    # Typical ML algorithm model builder workflow:
    # 1 Evaluate model with cross-validation methods: acc and auc.
    # 2 Choose best parameters for decision tree based on step 1 result.
    # 3 Train final model on full dataset.
    # 4 Run model on full data and build confusion matrix and report.
    def train(self, args: DtPeTrainAlgoArgs, data: DataFrame):
        ml_label = self.data_reader.get_label_as_series(data, args.label)
        ml_features = self.data_reader.get_features_data_frame(data, args.label)

        model = self.trainer.get_decision_tree_classifier(args)
        skf = self.trainer.get_split_train_test(args)
        scores = self.trainer.get_cross_validate_score(model, skf, ml_features, ml_label)
        model = self.trainer.fit_model(model, ml_features, ml_label)

        con_matrix = self.reporter.get_confusion_matrix(model, skf, ml_features, ml_label)
        return self.reporter.get_report(args, ml_features, model, con_matrix, scores)

