import numpy as np

from src.specific.lgb.trainer.pe_lgb_train_args import LgbPeTrainArgs
from src.specific.lgb.trainer.pe_lgb_train_result import LgbPeTrainResult


class LgbPeOutputMapper:
    """Map training results to output format."""

    def get_end_message(self, result: LgbPeTrainResult):
        """Generate markdown summary message from training results."""
        args = result.input_args
        parts = [
            "# LightGBM — Cross-Validation",
            "",
            f"- Splits: {args.n_splits}",
            f"- AUC (mean ± std): {np.mean(result.auc_score):.4f} ± {np.std(result.auc_score, ddof=1):.4f}",
            f"- Accuracy (mean ± std): {np.mean(result.acc_score):.4f} ± {np.std(result.acc_score, ddof=1):.4f}",
            f"- Samples: {result.input_features.shape[0]}, Features: {result.input_features.shape[1]}",
            f"- Model: n_estimators={args.n_estimators}, max_depth={args.max_depth}, learning_rate={args.learning_rate}",
            ""
        ]
        return "\n".join(parts)

    def get_report(self, args, result: LgbPeTrainResult):
        """Generate JSON-compatible report dictionary."""
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
        """Extract feature column names from results."""
        return result.input_features.columns.tolist()
