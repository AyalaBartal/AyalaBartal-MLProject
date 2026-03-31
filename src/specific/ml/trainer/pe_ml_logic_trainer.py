#!/usr/bin/env python3
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

from src.common.validator.args_validator import ArgsValidator
from src.specific.ml.trainer.pe_ml_report_trainer import MlPeReportTrainer
from src.specific.ml.trainer.pe_ml_data_trainer import MlPeDataTrainer
from src.specific.ml.trainer.pe_ml_model_trainer import MlPeModelTrainer
from src.specific.ml.trainer.pe_ml_train_algo_args import MlPeTrainAlgoArgs


class MlPeLogicTrainer:

    def __init__(self, reader: MlPeDataTrainer, trainer: MlPeModelTrainer, reporter: MlPeReportTrainer):
        ArgsValidator.require_type_not_none(reader, MlPeDataTrainer, "reader")
        ArgsValidator.require_type_not_none(trainer, MlPeModelTrainer, "trainer")
        ArgsValidator.require_type_not_none(reporter, MlPeReportTrainer, "reporter")
        self.data_reader = reader
        self.trainer = trainer
        self.reporter = reporter

    def train(self, args: MlPeTrainAlgoArgs, data: DataFrame):
        ml_label = self.data_reader.get_label_as_series(data, args.label)
        ml_features = self.data_reader.get_features_data_frame(data, args.label)

        model = self.trainer.get_mlp_model(ml_features.shape[1], args)
        skf = self.trainer.get_split_train_test(args)
        
        cv_results = self._cross_validate(model, skf, ml_features, ml_label, args)
        
        final_model = self.trainer.get_mlp_model(ml_features.shape[1], args)
        final_model = self.trainer.fit_model(final_model, ml_features, ml_label, args)

        return self.reporter.get_report(args, ml_features, final_model, cv_results)

    def _cross_validate(self, model, skf, ml_features, ml_label, args):
        """Custom cross-validation for PyTorch model"""
        test_auc_scores = []
        test_acc_scores = []
        con_matrices = []
        
        for train_idx, test_idx in skf.split(ml_features, ml_label):
            X_train, X_test = ml_features.iloc[train_idx], ml_features.iloc[test_idx]
            y_train, y_test = ml_label.iloc[train_idx], ml_label.iloc[test_idx]
            
            fold_model = self.trainer.get_mlp_model(ml_features.shape[1], args)
            fold_model = self.trainer.fit_model(fold_model, X_train, y_train, args)
            
            y_pred = self.trainer.predict(fold_model, X_test)
            y_pred_proba = self.trainer.predict_proba(fold_model, X_test)
            
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            acc = accuracy_score(y_test, y_pred)
            
            test_auc_scores.append(auc)
            test_acc_scores.append(acc)
            con_matrices.append(self.trainer.build(y_test, y_pred))
        
        return {
            'test_auc': np.array(test_auc_scores),
            'test_accuracy': np.array(test_acc_scores),
            'confusion_matrix': con_matrices
        }
