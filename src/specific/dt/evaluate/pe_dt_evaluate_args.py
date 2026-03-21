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
