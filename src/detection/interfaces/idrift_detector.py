from abc import ABC, abstractmethod
from typing import Union

from numpy import ndarray


class IDriftDetector(ABC):
    """
    Interface para a criação de classes responsáveis pela detecção do
    drift em grupos de instâncias.
    """
    def __init__(self):
        self.drift_counter = 0
        self.drift = False

    @abstractmethod
    def detect(self, chunk: Union[ndarray, float]) -> bool:
        """
        Identificador de uma detecção de drift que utiliza batches de
        instâncias.

        Parameters
        ----------
        chunk : ndarray or float
            Grupo de instâncias a ser submetido para a análise do
            detector. Pode utilizar de alguma métrica para validar
            o desempenho do comitê em termos de eficácia de
            classificação dos dados.

        Returns
        -------
        bool
            True se foi identificada uma mudança de contexto nos dados
            que estão sendo avaliados. Caso contrário, será retornado
            False.

        Raises
        ------
        NotImplementedError
            Requer sobreescrita nas classes filhas.
        """
        raise NotImplementedError("Método necessário não implementado!")

    def reset_params(self):
        """Retorna os parâmetros para o valor default."""
        self.drift_counter = 0
        self.drift = False

        return self

    def increase_counter(self):
        self.drift_counter += 1

    def __str__(self) -> str:
        msg = f'\nIdentificados {self.drift_counter} drifts na stream.\n'

        return msg
