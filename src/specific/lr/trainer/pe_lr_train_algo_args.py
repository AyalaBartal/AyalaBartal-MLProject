class LrPeTrainAlgoArgs:

    def __init__(self, solver='lbfgs', max_iter=1000, random_state=42, label='Label'):
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        self.label = label
        self.n_splits = 5
