import os


class DtPeEvaluateOutputArgs:

    def __init__(self, output_dir):
        self.output_dir = output_dir

        self.out_json = self.feature_importance_csv = os.path.join(output_dir, 'dt_test_metrics.json')
        self.out_md = self.feature_importance_csv = os.path.join(output_dir, 'd_test_summary.md')
        self.out_png = self.feature_importance_csv = os.path.join(output_dir, 'dt_confusion_matrix.png')
