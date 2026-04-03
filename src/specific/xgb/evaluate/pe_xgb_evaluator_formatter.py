class XgbPeEvaluatorFormatter:
    """Formatter for XGBoost evaluation results.
    
    Converts evaluation reports to JSON and Markdown formats.
    """

    @staticmethod
    def get_json_from_report_threshold(report, threshold: float) -> dict:
        """Format evaluation report as JSON.
        
        Args:
            report: XgbPeEvaluateReport instance.
            threshold: Classification threshold used.
            
        Returns:
            Dictionary with evaluation metrics in JSON format.
        """
        return {'auc': float(report.auc),
                'accuracy': float(report.acc),
                'threshold': threshold,
                'confusion_matrix': report.cm.tolist()
                }

    @staticmethod
    def get_md_from_report_threshold(report, threshold: float) -> str:
        """Format evaluation report as Markdown.
        
        Args:
            report: XgbPeEvaluateReport instance.
            threshold: Classification threshold used.
            
        Returns:
            Markdown formatted string with evaluation metrics.
        """
        lines = [
            "# XGBoost — Test Metrics",
            f"- AUC: {report.auc:.4f}",
            f"- Accuracy: {report.acc:.4f}",
            f"- Threshold: {threshold:.2f}",
            f"- Confusion: {report.cm.tolist()}",
        ]
        return "\n".join(lines)
