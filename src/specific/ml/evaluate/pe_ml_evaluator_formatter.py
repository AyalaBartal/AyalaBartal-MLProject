class MlPeEvaluatorFormatter:

    @staticmethod
    def get_json_from_report_threshold(report, threshold):
        return {'auc': float(report.auc),
                'accuracy': float(report.acc),
                'threshold': threshold,
                'confusion_matrix': report.cm.tolist()
                }

    @staticmethod
    def get_md_from_report_threshold(report, threshold):
        lines = [
            "# PyTorch MLP — Test Metrics",
            f"- AUC: {report.auc:.4f}",
            f"- Accuracy: {report.acc:.4f}",
            f"- Threshold: {threshold:.2f}",
            f"- Confusion: {report.cm.tolist()}",
        ]
        return "\n".join(lines)
