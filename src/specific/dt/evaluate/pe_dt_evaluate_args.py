import argparse
import os


class DtPeEvaluateArgs:

    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir

        self.input_csv = self.feature_importance_csv = os.path.join(input_dir, 'input.csv')
        self.column_label = 'Label'
        self.input_model = self.feature_importance_csv = os.path.join(input_dir, 'model')

        self.threshold = 0.5
        self.out_json = self.feature_importance_csv = os.path.join(output_dir, 'dt_test_metrics.json')
        self.out_md = self.feature_importance_csv = os.path.join(output_dir, 'd_test_summary.md')
        self.out_png = self.feature_importance_csv = os.path.join(output_dir, 'dt_confusion_matrix.png')



    @staticmethod
    def get_train_args_from_cmd():
        ap = argparse.ArgumentParser()
        ap.add_argument('--data', required=True)
        ap.add_argument('--label', required=True)
        ap.add_argument('--model', required=True)
        ap.add_argument('--threshold', type=float, default=0.5)
        ap.add_argument('--out-json', default='reports/dt_test_metrics.json')
        ap.add_argument('--out-md', default='reportst/d_test_summary.md')
        ap.add_argument('--out-cm-png', default='reports/dt_confusion_matrix.png')
        args = ap.parse_args()
        return args
