import os


class MlPeEvaluateOutputArgs:

    def __init__(self, output_dir):
        self.output_dir = output_dir

        self.out_json = os.path.join(output_dir, 'ml_test_metrics.json')
        self.out_md = os.path.join(output_dir, 'ml_test_summary.md')
        self.out_png = os.path.join(output_dir, 'ml_confusion_matrix.png')
