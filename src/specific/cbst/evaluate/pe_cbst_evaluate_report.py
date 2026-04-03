class CbstPeEvaluateReport:
    """Evaluation report containing computed metrics.
    
    Attributes:
        auc (float): Area Under the Receiver Operating Characteristic Curve.
        acc (float): Accuracy score.
        cm (ndarray): Confusion matrix.
    """

    def __init__(self, auc: float, acc: float, cm):
        """Initialize evaluation report.
        
        Args:
            auc: AUC score.
            acc: Accuracy score.
            cm: Confusion matrix as numpy array.
        """
        self.auc = auc
        self.acc = acc
        self.cm = cm
