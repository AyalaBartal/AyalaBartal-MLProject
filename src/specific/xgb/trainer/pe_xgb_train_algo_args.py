class XgbPeTrainAlgoArgs:
    """Algorithm arguments for XGBoost training."""

    def __init__(self):
        # Output column
        self.label = 'Label'

        # XGBClassifier hyperparameters
        self.n_estimators = 100
        self.max_depth = 6
        self.learning_rate = 0.1
        self.subsample = 0.8
        self.colsample_bytree = 0.8
        self.scale_pos_weight = 1.0
        self.random_state = 42
        self.n_splits = 10
