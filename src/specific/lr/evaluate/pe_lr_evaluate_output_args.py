import os


class LrPeEvaluateOutputArgs:

    def __init__(self, output_dir):
        self.output_dir = output_dir

        self.out_json = os.path.join(output_dir, 'lr_test_metrics.json')
        self.out_md = os.path.join(output_dir, 'lr_test_summary.md')
        self.out_png = os.path.join(output_dir, 'lr_confusion_matrix.png')
