from src.detection.interfaces.IChunk import IChunk


class FixedThreshold(IChunk):
    def __init__(self, threshold=0.8):
        super().__init__(threshold)

    def detect(self, chunk_acc) -> bool:
        if chunk_acc < threshold:
            self.increase_counter()

            return True
        else:
            self.drift = False

            return False

    def reset_params(self):
        self.drift_counter = 0
