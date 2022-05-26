from src.detection.interfaces.IDriftDetector import IDriftDetector


class IChunk(IDriftDetector):
    def __init__(self, threshold):
        super().__init__()
        self.detection_threshold = threshold
        self.last_chunk = None
        self.actual_chunk = None

    def update_chunks(self, labels):
        self.last_chunk = self.actual_chunk
        self.actual_chunk = labels
