class DtPeTrainResult:

    def __init__(self, input_args, input_features, dt_model, acc_score, auc_score, confusion_matrix):
        self.input_args = input_args
        self.input_features = input_features
        self.dt_model = dt_model
        self.acc_score = acc_score
        self.auc_score = auc_score
        self.confusion_matrix = confusion_matrix
