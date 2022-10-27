from numpy import ndarray

from src.detection.interfaces.drift_detector import DriftDetector


class Chunk(DriftDetector):
    """
    Interface para detectores de drift que utilizam as instâncias para
    identificar a ocorrência de drift na base de dados.
    """
    def __init__(self,):
        super().__init__()
        self.last_chunk = None
        self.actual_chunk = None
        self.detector_type = 'classes'

    def update_chunks(self, labels: ndarray) -> None:
        """
        Realiza a atualização das instâncias do chunk atual vão para o
        chunk anterior e as novas instâncias vão para o chunk atual.

        Parameters
        ----------
        labels : ndarray
            instâncias ou classes do chunk atual para serem atualizadas
        """
        self.last_chunk = self.actual_chunk
        self.actual_chunk = labels

    def is_valid_to_process(self) -> bool:
        if self.last_chunk is None or self.actual_chunk is None:
            return False
        return True

    def reset_params(self):
        super().reset_params()
        self.last_chunk = None
        self.actual_chunk = None

        return self
