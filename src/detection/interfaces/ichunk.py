from typing import NoReturn

from numpy import ndarray

from src.detection.interfaces.idrift_detector import IDriftDetector


class IChunk(IDriftDetector):
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
        self.last_chunk = None
        self.actual_chunk = None

    def update_chunks(self, labels: ndarray) -> NoReturn:
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

    def __str__(self) -> str:
        msg = super().__str__()
        msg += f'Threshold é de {self.detection_threshold}.\n'
        return 
