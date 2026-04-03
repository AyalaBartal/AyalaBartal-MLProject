class CbstPeTrainAlgoArgs:
    """Algorithm arguments for CatBoost training."""

    def __init__(self):
        # Output column
        self.label = 'Label'

        # CatBoostClassifier hyperparameters
        self.n_estimators = 100
        self.max_depth = 6
        self.learning_rate = 0.1
        self.random_state = 42
        self.n_splits = 5
