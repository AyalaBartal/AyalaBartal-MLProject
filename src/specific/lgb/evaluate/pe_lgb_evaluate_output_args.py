import os


class LgbPeEvaluateOutputArgs:
    """Output arguments for LightGBM evaluator.
    
    Specifies paths to output directory and result files.
    
    Attributes:
        output_dir (str): Path to directory for output files.
        out_json (str): Path to JSON file for evaluation metrics.
        out_md (str): Path to Markdown file for evaluation summary.
        out_png (str): Path to PNG file for confusion matrix plot.
    """

    def __init__(self, output_dir: str):
        """Initialize output arguments.
        
        Args:
            output_dir: Base directory for output files.
        """
        self.output_dir = output_dir

        self.out_json = os.path.join(output_dir, 'lgb_test_metrics.json')
        self.out_md = os.path.join(output_dir, 'lgb_test_summary.md')
        self.out_png = os.path.join(output_dir, 'lgb_confusion_matrix.png')
