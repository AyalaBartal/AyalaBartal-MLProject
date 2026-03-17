import os


class DtPeTrainArgs:

    def __init__(self, input_csv, out_report_dir):
        self.input_csv = input_csv

        self.out_report_md = os.path.join(out_report_dir, 'dt_cv_summary.md')
        self.out_report_json = os.path.join(out_report_dir, 'dt_cv_metrics.json')
        self.out_schema_json = os.path.join(out_report_dir, 'dt_feature_schema.json')
        self.out_model_joblib = os.path.join(out_report_dir, 'decision_tree_model.joblib')
        self.model_output_dot = os.path.join(out_report_dir, 'decision_tree_model.dot')
        self.feature_importance_csv = os.path.join(out_report_dir, 'feature_importance.csv')

        self.label = 'Label'
        self.criterion = 'gini'

        self.n_splits = 3
        self.random_state = 42
        self.max_depth = 24
        self.min_samples_leaf = 2
