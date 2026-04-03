class LgbPeEvaluateAlgoArgs:
    """Algorithm-specific evaluation arguments for LightGBM evaluator.
    
    Attributes:
        column_label (str): Name of the label column in the input data.
        threshold (float): Probability threshold for binary classification predictions.
    """

    def __init__(self):
        self.column_label = 'Label'
        self.threshold = 0.5
