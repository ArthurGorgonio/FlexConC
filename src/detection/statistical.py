from typing import Any, Dict
from scipy.stats import kstest

from src.detection.interfaces.statisticaltest import StatisticalTest


class Statistical(StatisticalTest):
    """
    Detector de drift que usa testes estatísticos para determinar se
    houve mudança de contexto entre o chunk anterior e o atual.

    Parameters
    ----------
    threshold : float
        _description_
    """
    def __init__(self, **params: Dict[str, Any]):
        super().__init__()
        if params is None:
            params = {
                "alpha": 0.05,
                "statistical_test": kstest,
            }
        self.alpha = params.get("alpha") or 0.05
        self.statistical_test = params.get("statistical_test") or kstest

    def eval_test(self):
        return self.statistical_test(
            self.actual_chunk,
            self.last_chunk
        ).pvalue if self.is_valid_to_process() else 1.0

    def detect(self, chunk):
        self.update_chunks(chunk)
        p_value = self.eval_test()

        if p_value <= self.alpha:
            self.increase_counter()
            self.drift = True

            return self.drift
        self.drift = False
        return self.drift

    def reset_params(self):
        super().reset_params()
