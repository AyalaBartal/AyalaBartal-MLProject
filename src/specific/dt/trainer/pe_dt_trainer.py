#!/usr/bin/env python3

from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix

from src.specific.dt.trainer.pe_dt_train_algo_args import DtPeTrainAlgoArgs
from src.specific.dt.trainer.pe_dt_train_args import DtPeTrainArgs
from src.specific.dt.trainer.pe_dt_train_result import DtPeTrainResult


class DtPeDataTrainer:

    def train(self, args: DtPeTrainAlgoArgs, data: DataFrame):
        # (1) Validate input
        self.validate_train_input(args, data)

        # (2) A decision tree classifier
        model = self.get_decision_tree_classifier(args)
        # Provides train/test indices to split data in train/test sets
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)

        # (3) Evaluate a score by cross-validation.
        ml_label = data[args.label]  # Series
        ml_features = data.drop(columns=[args.label])
        auc = cross_val_score(model, ml_features, ml_label, cv=skf, scoring='roc_auc', n_jobs=-1)
        acc = cross_val_score(model, ml_features, ml_label, cv=skf, scoring='accuracy', n_jobs=-1)

        # (4) Build a decision tree classifier from the training set (ml_features, ml_label).
        model.fit(ml_features, ml_label)

        con_matrix = self.calc_confusion_matrix(ml_features, ml_label, model, skf)

        return DtPeTrainResult(args, ml_features, model, acc, auc, con_matrix)


    def validate_train_input(self, args, data):
        if args is None:
            raise ValueError("args cannot be None")
        if not isinstance(args, DtPeTrainAlgoArgs):
            raise TypeError("args must be instance of DtPeTrainAlgoArgs")
        if data is None:
            raise ValueError("data cannot be None")
        if not isinstance(data, DataFrame):
            raise TypeError("data must be instance of DataFrame")

    def get_decision_tree_classifier(self, args):
        return DecisionTreeClassifier(criterion=args.criterion,
                                      max_depth=args.max_depth,
                                      min_samples_leaf=args.min_samples_leaf,
                                      class_weight='balanced',
                                      random_state=args.random_state)

    def calc_confusion_matrix(self, x, y, model, skf):
        all_true = []
        all_pred = []
        for train_idx, test_idx in skf.split(x, y):
            X_train, X_test = x.iloc[train_idx], x.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            all_true.extend(y_test)
            all_pred.extend(y_pred)
        return confusion_matrix(all_true, all_pred)
