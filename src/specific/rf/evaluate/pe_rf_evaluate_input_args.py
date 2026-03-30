import os


class RfPeEvaluateInputArgs:

    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.input_csv = os.path.join(input_dir, 'input.csv')
        self.input_model = os.path.join(input_dir, 'model')
