from abc import abstractmethod

from src.detection.interfaces.chunk import Chunk


class StatisticalTest(Chunk):
    def __init__(self):
        super().__init__()
        self.statistical_test = "kolmogorov"

    @abstractmethod
    def eval_test(self) -> float:
        """
        Realiza um testes estatístico nas instâncias dos chunks atual e
        anterior para validar se houve alguma mudança de contexto na
        base de dados.

        Returns
        -------
        float
            Executa o teste estatístico e retorna a métrica p_value.

        Raises
        ------
        NotImplementedError
            Requer sobreescrita nas classes filhas.
        """
        raise NotImplementedError("Método necessário não implementado!")
