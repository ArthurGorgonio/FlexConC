from typing import Any, Dict

from src.detection.interfaces.threshold import Threshold


class Normal(Threshold):
    """
    Detector de drift que utiliza a predição do comitê de
    classificadores para determinar o próximo threshold.
    """

    def __init__(self, **params: Dict[str, Any]):
        super().__init__(**params)
        self.change_thr = False

    def detect(self, chunk, y_pred) -> bool:
        self.__threshold_check()

        if chunk < self.detection_threshold:
            self.increase_counter()
            self.drift = True
            self.change_thr = True

            return self.drift
        self.drift = False

        return self.drift

    def __threshold_check(self, chunk):
        if self.change_thr:
            self.change_thr = False
            self.__calculate_new_threshold(chunk)

    def __calculate_new_threshold(self, chunk: float) -> None:
        if chunk == 0.0:
            self.detection_threshold = chunk + 0.01
        elif chunk == 1.0:
            self.detection_threshold = chunk - 0.01
        else:
            self.detection_threshold = chunk
