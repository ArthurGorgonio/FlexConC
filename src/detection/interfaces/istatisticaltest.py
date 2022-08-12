from abc import abstractmethod

from src.detection.interfaces.ichunk import IChunk


class IStatisticalTest(IChunk):
    def __init__(self, threshold):
        super().__init__(threshold)
        self.statistical_test = "kolmogorov"

    @abstractmethod
    def eval_test(self, alpha):
        ...
