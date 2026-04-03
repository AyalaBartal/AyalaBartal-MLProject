import os


class XgbPeTrainArgs:
    """General training arguments and output paths."""

    def __init__(self, input_csv, out_report_dir):
        self.input_csv = input_csv

        self.out_report_md = os.path.join(out_report_dir, 'xgb_cv_summary.md')
        self.out_report_json = os.path.join(out_report_dir, 'xgb_cv_metrics.json')
        self.out_schema_json = os.path.join(out_report_dir, 'xgb_feature_schema.json')
        self.out_model_joblib = os.path.join(out_report_dir, 'xgboost_model.joblib')
        self.feature_importance_csv = os.path.join(out_report_dir, 'feature_importance.csv')

        self.label = 'Label'

        # XGBClassifier hyperparameters
        self.n_estimators = 100
        self.max_depth = 6
        self.learning_rate = 0.1
        self.subsample = 0.8
        self.colsample_bytree = 0.8
        self.scale_pos_weight = 1.0
        self.random_state = 42
        self.n_splits = 10
