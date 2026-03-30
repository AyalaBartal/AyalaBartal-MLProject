class MlPeTrainOutputMapper:

    def map_report_to_output(self, report, train_result):
        metrics = {
            'cv_auc_mean': float(report['cv_auc_mean']),
            'cv_auc_std': float(report['cv_auc_std']),
            'cv_accuracy_mean': float(report['cv_accuracy_mean']),
            'cv_accuracy_std': float(report['cv_accuracy_std']),
            'confusion_matrix': report['confusion_matrix']
        }
        feature_schema = {
            'feature_order': report['feature_names'],
            'n_features': len(report['feature_names'])
        }
        return {
            'model': report['model'],
            'metrics': metrics,
            'feature_schema': feature_schema
        }
