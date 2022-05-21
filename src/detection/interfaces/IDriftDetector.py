from abc import ABC, abstractmethod

class IDriftDetector(ABC):
    def __init__(self):
        self.drift = False
        self.drift_counter = 0

    @abstractmethod
    def detect(self, chunk):
        ...

    @abstractmethod
    def reset_params(self):
        ...

    def increase_counter(self):
        self.drift_counter += 1
