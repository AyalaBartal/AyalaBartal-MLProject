class CbstPeTrainResult:
    """Result object containing training outcomes."""

    def __init__(self, input_args, input_features, cbst_model, confusion_matrix, acc_score, auc_score):
        self.input_args = input_args
        self.input_features = input_features
        self.cbst_model = cbst_model
        self.confusion_matrix = confusion_matrix
        self.acc_score = acc_score
        self.auc_score = auc_score
