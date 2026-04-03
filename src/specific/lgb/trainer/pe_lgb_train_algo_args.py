class LgbPeTrainAlgoArgs:
    """Algorithm arguments for LightGBM training."""

    def __init__(self):
        # Output column
        self.label = 'Label'

        # LGBMClassifier hyperparameters
        self.n_estimators = 100
        self.max_depth = 6
        self.learning_rate = 0.1
        self.num_leaves = 31
        self.random_state = 42
        self.n_splits = 10
