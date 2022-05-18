from src.detection.interfaces.IChunk import IChunk


class IStatisticalTest(IChunk):
    def __init__(self):
        self.statistical_test = "kolmogorov"

    def eval_test(self, alpha):
        pass
