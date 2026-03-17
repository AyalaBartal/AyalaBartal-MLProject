class DtPeTrainAlgoArgs:

    def __init__(self):
        # Output column
        self.label = 'Label'

        # DecisionTreeClassifier arg criterion define how to measure the quality of a split at each node.
        self.criterion = 'gini'

        # DecisionTreeClassifier args
        self.n_splits = 3
        self.random_state = 42
        self.max_depth = 24
        self.min_samples_leaf = 2
