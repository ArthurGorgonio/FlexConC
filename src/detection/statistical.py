from typing import Any, Dict

from numpy import ndarray, where
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
        self.detection_threshold = 0.8

        if params is None:
            params = {
                "alpha": 0.05,
                "statistical_test": kstest,
            }
        self.alpha = params.get("alpha") or 0.05
        self.statistical_test = params.get("statistical_test") or kstest

    def eval_test(self):
        return (
            self.statistical_test(self.actual_chunk, self.last_chunk).pvalue
            if self.is_valid_to_process()
            else 1.0
        )

    def detect(self, chunk, y_pred) -> bool:
        self.update_chunks(self.__compare_labels(chunk, y_pred))
        p_value = self.eval_test()

        if p_value <= self.alpha:
            self.increase_counter()
            self.drift = True

            return self.drift
        self.drift = False

        return self.drift

    def __compare_labels(self, y_true: ndarray, y_pred: ndarray) -> ndarray:
        """Função que computa quantos rótulos do chunk estão corretos.

        Args:
        ----
            - y_true: rótulos verdadeiros.
            - y_pred: rótulos preditos.

        Returns:
        -------
            array preenchido com 1 quando as classes convergem e 0
            quando divergem.
        """
        return where((y_true == y_pred), 1, 0)
