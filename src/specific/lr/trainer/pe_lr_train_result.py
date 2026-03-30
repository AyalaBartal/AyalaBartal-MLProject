class LrPeTrainResult:

    def __init__(self, status, message, model=None, metrics=None):
        self.status = status
        self.message = message
        self.model = model
        self.metrics = metrics
