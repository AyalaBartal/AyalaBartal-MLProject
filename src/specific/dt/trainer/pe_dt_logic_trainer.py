#!/usr/bin/env python3
from pandas import DataFrame

from src.specific.dt.trainer import DtPeReportTrainer
from src.specific.dt.trainer.pe_dt_data_trainer import DtPeDataTrainer
from src.specific.dt.trainer.pe_dt_model_trainer import DtPeModelTrainer
from src.specific.dt.trainer.pe_dt_train_algo_args import DtPeTrainAlgoArgs


class DtPeLogicTrainer:

    def __init__(self, reader: DtPeDataTrainer, trainer: DtPeModelTrainer, reporter: DtPeReportTrainer):
        self.data_reader = reader
        self.trainer = trainer
        self.reporter = reporter

    # Typical ML algorithm model builder workflow:
    # 1 Evaluate model with cross-validation methods: acc and auc.
    # 2 Choose best parameters for decision tree based on step 1 result.
    # 3 Train final model on full dataset.
    # 4 Run model on full data and build confusion matrix and report.
    def train(self, args: DtPeTrainAlgoArgs, data: DataFrame):
        self.validate_train_input(args, data)

        ml_label = self.data_reader.get_label_as_series(data, args.label)
        ml_features = self.data_reader.get_features_data_frame(data, args.label)

        model = self.trainer.get_decision_tree_classifier(args)
        skf = self.trainer.get_split_train_test(args)
        scores = self.trainer.get_cross_validate_score(model, skf, ml_features, ml_label)
        model = self.trainer.fit_model(model, ml_features, ml_label)

        con_matrix = self.reporter.get_confusion_matrix(model, skf, ml_features, ml_label)
        return self.reporter.get_report(args, ml_features, model, con_matrix, scores)

    def validate_train_input(self, args, data):
        if args is None:
            raise ValueError("args cannot be None")
        if not isinstance(args, DtPeTrainAlgoArgs):
            raise TypeError("args must be instance of DtPeTrainAlgoArgs")
        if data is None:
            raise ValueError("data cannot be None")
        if not isinstance(data, DataFrame):
            raise TypeError("data must be instance of DataFrame")
