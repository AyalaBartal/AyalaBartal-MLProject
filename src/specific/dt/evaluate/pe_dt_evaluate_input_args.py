import os


class DtPeEvaluateInputArgs:

    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.input_csv = self.feature_importance_csv = os.path.join(input_dir, 'input.csv')
        self.input_model = self.feature_importance_csv = os.path.join(input_dir, 'model')
