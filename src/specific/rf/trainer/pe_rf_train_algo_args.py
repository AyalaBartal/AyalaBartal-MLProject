class RfPeTrainAlgoArgs:

    def __init__(self):
        # Output column
        self.label = 'Label'

        # RandomForestClassifier args
        self.n_estimators = 100
        self.max_depth = 20
        self.random_state = 42
        self.n_splits = 10
