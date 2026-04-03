import os


class XgbPeEvaluateInputArgs:
    """Input arguments for XGBoost evaluator.
    
    Specifies paths to input directory, CSV data file, and model file.
    
    Attributes:
        input_dir (str): Path to directory containing input files.
        input_csv (str): Path to CSV file with evaluation data.
        input_model (str): Path to XGBoost model file.
    """

    def __init__(self, input_dir: str):
        """Initialize input arguments.
        
        Args:
            input_dir: Base directory containing input files.
        """
        self.input_dir = input_dir
        self.input_csv = os.path.join(input_dir, 'input.csv')
        self.input_model = os.path.join(input_dir, 'model')
