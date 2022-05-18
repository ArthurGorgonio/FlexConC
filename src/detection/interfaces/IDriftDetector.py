class IDriftDetector:
    def __init__(self):
        self.drift = False
        self.drift_counter = 0

    def detect(self, X):
        raise NotImplementedError

    def reset_params(self):
        raise NotImplementedError

    def increase_counter(self):
        self.drift_counter += 1
