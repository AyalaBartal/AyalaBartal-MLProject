import argparse
import os


class DtPeTrainArgs:

    def __init__(self, input_csv, out_report_dir, out_model):
        self.input_csv = input_csv
        self.out_model = out_model

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

    @staticmethod
    def get_train_args_from_cmd():
        ap = argparse.ArgumentParser()
        ap.add_argument('--input-csv', required=True)
        ap.add_argument('--label', required=True)
        ap.add_argument('--n-splits', type=int, default=5)
        ap.add_argument('--random-state', type=int, default=42)
        ap.add_argument('--max-depth', type=int, default=None)
        ap.add_argument('--min-samples-leaf', type=int, default=1)
        ap.add_argument('--criterion', default='gini', choices=['gini', 'entropy', 'log_loss'])
        ap.add_argument('--out-model', default='models/dt_model.joblib')
        ap.add_argument('--out-report-json', default='reports/dt_cv_metrics.json')
        ap.add_argument('--out-report-md', default='reports/dt_cv_summary.md')
        ap.add_argument('--out-schema-json', default='models/dt_feature_schema.json')
        args = ap.parse_args()
        return args
