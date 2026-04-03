class CbstPeEvaluateAlgoArgs:
    """Algorithm-specific evaluation arguments for CatBoost evaluator.
    
    Attributes:
        column_label (str): Name of the label column in the input data.
        threshold (float): Probability threshold for binary classification predictions.
    """

    def __init__(self):
        self.column_label = 'Label'
        self.threshold = 0.5
