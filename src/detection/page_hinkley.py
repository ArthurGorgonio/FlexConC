"""
Based on source code of:
https://github.com/blablahaha/concept-drift/blob/master/concept_drift/page_hinkley.py

"""

from src.detection.interfaces.IDriftDetector import IDriftDetector


class PageHinkley(IDriftDetector):
    def __init__(
        self,
        delta: float = 5e-2,
        lambda_: int = 50,
        alpha: float = 0.9999
    ):
        super().__init__()
        self.delta = delta
        self.lambda_ = lambda_
        self.alpha = alpha
        self.sum = 0
        self.x_mean = 0
        self.num = 0
        self.change_detected = False

    def __reset_params(self):
        """
        Every time a change has been detected, all the collected statistics are reset.
        :return:
        """
        self.num = 0
        self.x_mean = 0
        self.sum = 0

    def set_input(self, x) -> bool:
        """
        Main method for adding a new data value and automatically detect
        a possible concept drift.

        Args:
            x: instances

        Returns:
            bool: a drift detection
        """
        self.__detect_drift(x)
        return self.change_detected

    def __detect(self, x):
        """
        Concept drift detection following the formula from 'Knowledge Discovery from Data Streams' by João Gamma (p. 76)

        Args:
            x: instances
        """
        # calculate the average and sum
        self.num += 1
        self.x_mean = (x + self.x_mean * (self.num - 1)) / self.num
        self.sum = self.sum * self.alpha_ + (x - self.x_mean - self.delta_)

        self.change_detected = True if self.sum > self.lambda_ else False
        if self.change_detected:
            self.__reset_params()