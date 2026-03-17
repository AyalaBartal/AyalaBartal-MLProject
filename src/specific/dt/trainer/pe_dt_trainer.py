#!/usr/bin/env python3
import json, os

import joblib
import numpy as np, pandas as pd
from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix
from joblib import dump

from src.specific.dt.trainer.pe_dt_train_args import DtPeTrainArgs
from src.specific.dt.trainer.pe_dt_train_result import DtPeTrainResult


class DtPeDataTrainer:

    def train(self, args: DtPeTrainArgs, data: DataFrame):
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

        result = DtPeTrainResult(args, ml_features, model, acc, auc, con_matrix)

        # (5) Write results into output files
        self.write_output(result)

    def validate_train_input(self, args, data):
        if args is None:
            raise ValueError("args cannot be None")
        if not isinstance(args, DtPeTrainArgs):
            raise TypeError("args must be instance of DtPeTrainArgs")
        if data is None:
            raise ValueError("data cannot be None")
        if not isinstance(data, DataFrame):
            raise TypeError("data must be instance of DataFrame")

    def write_output(self, result: DtPeTrainResult):
        args = result.input_args
        md = self.get_message(args, result.input_features, result.acc_score, result.auc_score)
        print(md)

        dump(result.dt_model, args.out_model)

        # Save schema JSON
        with open(args.out_schema_json, 'w', encoding='utf-8') as f:
            json.dump({'feature_order': list(result.input_features.columns)}, f)

        # Save report JSON
        # get_report(self, args, ml_features, acc, auc, cm):
        rep = self.get_report(args, result.input_features, result.acc_score, result.auc_score, result.confusion_matrix)
        with open(args.out_report_json, 'w', encoding='utf-8') as f:
            json.dump(rep, f, indent=2)

        # Save markdown report
        with open(args.out_report_md, 'w', encoding='utf-8') as f:
            f.write(md)

        joblib.dump(result.dt_model, args.out_model_joblib)

        # Optional: export tree visualization (.dot)
        model_output_dot = args.model_output_dot
        feature_names = result.input_features.columns.tolist()
        export_graphviz(
            result.dt_model,
            out_file=str(model_output_dot),
            feature_names=feature_names,
            class_names=[str(c) for c in result.dt_model.classes_],
            filled=True,
            rounded=True
        )

        # --------------------
        # Feature Importance
        # --------------------
        feature_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": result.dt_model.feature_importances_
        }).sort_values(by="importance", ascending=False)
        feature_importance.to_csv(args.feature_importance_csv, index=False)


    def get_report(self, args, ml_features, acc, auc, cm):
        return {
            'cv_splits': args.n_splits,
            'cm': cm.tolist(),
            'auc_mean': float(np.mean(auc)),
            'auc_std': float(np.std(auc, ddof=1)),
            'acc_mean': float(np.mean(acc)),
            'acc_std': float(np.std(acc, ddof=1)),
            'n_features': int(ml_features.shape[1]),
            'n_samples': int(ml_features.shape[0])
        }


    def get_message(self,args, ml_features, acc, auc):
        return f"""# Decision Tree — Cross-Validation\n\n- Splits: {args.n_splits}\n- AUC (mean ± std): {np.mean(auc):.4f} ± {np.std(auc, ddof=1):.4f}\n- Accuracy (mean ± std): {np.mean(acc):.4f} ± {np.std(acc, ddof=1):.4f}\n- Samples: {ml_features.shape[0]}, Features: {ml_features.shape[1]}\n- Model: criterion={args.criterion}, max_depth={args.max_depth}, min_samples_leaf={args.min_samples_leaf}\n"""


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
