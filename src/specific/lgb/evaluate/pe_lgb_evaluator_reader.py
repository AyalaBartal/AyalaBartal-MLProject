import pandas as pd
from joblib import load

from src.common.validator.file_validator import FileValidator


class LgbPeEvaluatorReader:
    """Reader for LightGBM model and evaluation data.
    
    Handles reading LightGBM models from joblib files and CSV data.
    Provides validation for input and output directories.
    """

    def __init__(self, validator: FileValidator):
        """Initialize reader with directory validator.
        
        Args:
            validator: FileValidator instance for directory validation.
        """
        self.validator = validator

    def validate_output(self, args_out):
        """Validate output directory exists and is writable.
        
        Args:
            args_out: Output arguments containing output_dir path.
            
        Raises:
            Exception: If directory validation fails.
        """
        self.validator.validate_directory(args_out.output_dir, True, True)

    def validate_input(self, args_in):
        """Validate input directory exists and is readable.
        
        Args:
            args_in: Input arguments containing input_dir path.
            
        Raises:
            Exception: If directory validation fails.
        """
        self.validator.validate_directory(args_in.input_dir, True, False)

    def read_lgb_model_from_joblib_file(self, args_in):
        """Read LightGBM model from joblib file.
        
        Args:
            args_in: Input arguments containing input_model path.
            
        Returns:
            Loaded LightGBM model.
        """
        return load(args_in.input_model)

    def read_csv_to_df(self, input_csv: str) -> pd.DataFrame:
        """Read CSV file into DataFrame.
        
        Args:
            input_csv: Path to CSV file.
            
        Returns:
            DataFrame containing CSV data.
        """
        return pd.read_csv(input_csv)
