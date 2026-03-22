class DtPeEvaluatorFormatter:

    @staticmethod
    def get_json_from_report_threshold(report, threshold):
        return {'auc': float(report.auc),
                'accuracy': float(report.acc),
                'threshold': threshold,
                'confusion_matrix': report.cm.tolist()
                }

    def get_md_from_report_threshold(self, report, threshold):
        acc = report.acc
        auc = report.auc
        cm = report.cm
        md = f"""# Decision Tree — Test Metrics\n\n- AUC: {auc:.4f}\n- Accuracy: {acc:.4f}\n- Threshold: {threshold:.2f}\n\n**Confusion Matrix** (rows=Actual, cols=Predicted):\n{cm.tolist()}\n"""
        return md
