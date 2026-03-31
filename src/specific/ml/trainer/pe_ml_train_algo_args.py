class MlPeTrainAlgoArgs:

    def __init__(self, hidden_sizes=None, learning_rate=0.001, epochs=100, batch_size=32, 
                 random_state=42, label='Label'):
        self.hidden_sizes = hidden_sizes if hidden_sizes else [128, 64, 32]
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.label = label
        self.n_splits = 5
