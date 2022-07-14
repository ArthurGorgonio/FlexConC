from abc import abstractmethod

from src.ssl.ensemble import Ensemble


class IReaction:
    def __init__(self):
        pass

    @abstractmethod
    def react(self, ensemble, instances, labels) -> Ensemble:
        """
        Método para realizar uma reação ao drift, a reação depende como
        como cada uma das classes filhas irá implementar a reação.
        Todas as classes irão modificar a estrutura do comitê.

        Args:
            ensemble: Comitê de classificadores.
            instances: instâncias que serão utilizadas no treinamento.
            labels: Rótulo das instâncias.

        Raises:
            NotImplementedError: Caso a classe herde e não implemente
            esse método será apresentado o erro NotImplementedError.

        Return:
            Um novo comitê, que está apto a realizar a classificação de
            novos dados.
        """
        raise NotImplementedError(
            "As classes filhas devem implementar esse método."
        )
