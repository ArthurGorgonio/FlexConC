from typing import Any, Dict


def validate_estimator(estimator):
    """Make sure that an estimator implements the necessary methods."""
    if not hasattr(estimator, "predict_proba"):
        msg = "base_estimator ({}) should implement predict_proba!"
        raise ValueError(msg.format(type(estimator).__name__))


class Log():
    """
    Classe para gerenciar o Log que é gerado num arquivo txt com
    informações detalhadas da iteração do método.

    Parameters
    ----------
    filename : str
        Nome do arquivo gerado. Essa info é compartilhada entre as
        instâncias da classe.
    """
    _instance = None
    filename = 'output.txt'

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def write_archive_header(self):
        """Função para escrever os cabeçalhos dos arquivos"""
        infos = [f'info0{i}'.rjust(10) for i in range(1,9)]
        header = (
            'In each iteration, info means:\n'
            'info01: ensemble size;\n'
            'info02: ensemble hits;\n'
            'info03: ensemble detect a drift;\n'
            'info04: Total processed instances;\n'
            'info05: ensemble Acc;\n'
            'info06: ensemble F-Measure;\n'
            'info07: ensemble kappa statistics;\n'
            'info08: Elapsed iteration time.\n'
            + '-' * 88
            + '\n'
            + ' '.join(infos)
            + '\n'
            + '-' * 88
            + '\n'
        )

        self.write_archive(header)

    def write_archive_output(self, **kargs: Dict[str, Any]):
        """
        Função responsável por escrever o log de uma iteração com o
        dicionário que é passado por parâmetro para o método.

        Essa dicionário deve ter a seguinte estrutura:
        >>> {
        ...     'ensemble_size': 10,
        ...     'ensemble_hits': 100,
        ...     'drift_detected': False,
        ...     'instances': 400,
        ...     'enlapsed_time': 3213.321,
        ...     'metrics': {
        ...         'acc': 0.99,
        ...         'f1': 0.89,
        ...         'kappa': 0.90,
        ...     },
        ... }

        Parameters
        ----------
        kargs : Dict[str, Any]
            Dicionário com as informações a serem colocadas no arquivo.
        """
        infos = (
            f"{kargs['ensemble_size']}".rjust(11)
            + f"{kargs['ensemble_hits']}".rjust(11)
            + f"{kargs['drift_detected']}".rjust(11)
            + f"{kargs['instances']}".rjust(11)
            + f"{kargs['metrics']['acc']}".rjust(11)
            + f"{kargs['metrics']['f1']}".rjust(11)
            + f"{kargs['metrics']['kappa']}".rjust(11)
            + f"{kargs['enlapsed_time']}".rjust(11)
            + "\n"
        )

        self.write_archive(infos)

    def write_archive(self, *args):
        """
        Função para abstrair a escrita no arquivo.

        Parameters
        ----------
        args : Any
            Conteúdo a ser escrito no arquivo.
        """
        with open(self.filename, 'a') as f:
            f.writelines(args)
