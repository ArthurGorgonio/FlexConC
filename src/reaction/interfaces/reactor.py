from abc import abstractmethod

from numpy import ndarray

from src.ssl.ensemble import Ensemble


class Reactor:
    @abstractmethod
    def react(
        self,
        ensemble: Ensemble,
        instances: ndarray,
        labels: ndarray
    ) -> Ensemble:
        """
        Método para realizar uma reação ao drift, a reação depende como
        como cada uma das classes filhas irá implementar a reação.
        Todas as classes irão modificar a estrutura do comitê.

        Parameters
        ----------
        ensemble : Ensemble
            Comitê de classificadores.
        instances : ndarray
            instâncias que serão utilizadas no treinamento.
        labels : ndarray
            Rótulo das instâncias.

        Returns
        -------
        Ensemble
            Um novo comitê, que está apto a realizar a classificação de
            novos dados.

        Raises
        ------
            NotImplementedError: Caso a classe herde e não implemente
            esse método será apresentado o erro NotImplementedError.
        """
        raise NotImplementedError(
            "As classes filhas devem implementar esse método."
        )
