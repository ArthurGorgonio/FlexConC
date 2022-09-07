from numpy import ndarray

from src.detection.interfaces.drift_detector import DriftDetector


class Threshold(DriftDetector):
    """
    Interface para detectores de drift que utilizam as instâncias para
    identificar a ocorrência de drift na base de dados.

    Parameters
    ----------
    threshold : float
        limiar para a detecção do drift. É esperado valores entre
        [0, 1].
    """
    def __init__(self, threshold: float = 0.8):
        super().__init__()
        self.detection_threshold = threshold
        self.default_threshold = threshold
        self.detector_type = 'metric'

    def __str__(self) -> str:
        msg = super().__str__()
        msg += f'Threshold é de {self.detection_threshold}.\n'
        return msg

    def reset_params(self):
        super().reset_params()
        self.detection_threshold = self.default_threshold

        return self