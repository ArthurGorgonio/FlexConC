from src.detection.interfaces.IDriftDetector import IDriftDetector


class IChunk(IDriftDetector):
    def __init__(self):
        self.detection_threshold = None
        self.last_chunk = None
        self.actual_chunk = None

    def update_chunks(self, labels):
        self.last_chunk = self.actual_chunk
        self.actual_chunk = labels
