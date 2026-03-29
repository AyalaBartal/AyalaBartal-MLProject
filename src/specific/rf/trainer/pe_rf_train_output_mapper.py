import numpy as np

from src.specific.rf.trainer.pe_rf_train_args import RfPeTrainArgs
from src.specific.rf.trainer.pe_rf_train_result import RfPeTrainResult


class RfPeTrainOutputMapper:

    def get_end_message(self, result: RfPeTrainResult):
        args = result.input_args
        parts = [
            "# Random Forest — Cross-Validation",
            "",
            f"- Splits: {args.n_splits}",
            f"- AUC (mean ± std): {np.mean(result.auc_score):.4f} ± {np.std(result.auc_score, ddof=1):.4f}",
            f"- Accuracy (mean ± std): {np.mean(result.acc_score):.4f} ± {np.std(result.acc_score, ddof=1):.4f}",
            f"- Samples: {result.input_features.shape[0]}, Features: {result.input_features.shape[1]}",
            f"- Model: n_estimators={args.n_estimators}, max_depth={args.max_depth}",
            ""
        ]
        return "\n".join(parts)

    def get_report(self, args, result: RfPeTrainResult):
        return {
            'cv_splits': args.n_splits,
            'cm': result.confusion_matrix.tolist(),
            'auc_mean': float(np.mean(result.auc_score)),
            'auc_std': float(np.std(result.auc_score, ddof=1)),
            'acc_mean': float(np.mean(result.acc_score)),
            'acc_std': float(np.std(result.acc_score, ddof=1)),
            'n_features': int(result.input_features.shape[1]),
            'n_samples': int(result.input_features.shape[0])
        }

    def get_feature_columns_list(self, result):
        return result.input_features.columns.tolist()
