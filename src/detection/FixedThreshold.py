from src.detection.interfaces.IChunk import IChunk


class FixedThreshold(IChunk):
    def __init__(self, threshold):
        self.threshold = threshold

    def detect(self, chunk):
        ...

    def reset_params(self):
        ...
