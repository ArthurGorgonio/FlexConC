from scipy.stats import kstest

from src.detection.interfaces.IStatisticalTest import IStatisticalTest


class WeightedStatistical(IStatisticalTest):
    def __init__(self, threshold):
        super().__init__(threshold)
        self.sig_level = 1
    
    def eval_test(self, alpha: float = 0.05):
        return kstest(self.actual_chunk, self.last_chunk).pvalue

    def detect(self, chunk):
        ...

    def reset_params(self):
        ...