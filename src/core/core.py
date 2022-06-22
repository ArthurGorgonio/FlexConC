from src.detection import (
    cpssds,
    fixed_threshold,
    normal,
    page_hinkley,
    weighted,
    weighted_statistical
)
from src.reaction import Exchange
from src.ssl.ensemble import Ensemble


class Core:
    _base_classifiers = None
    def __init__(self):
        self.ensemble = None
        self.reaction = None
        self.detection = None

    def configure_params(
        self,
        ssl_algorithm,
        detector,
        reactor,
        params_detector: dict[str, any] = {},
        params_reactor: dict[str, any] = {}
    ):
        """
        Função que configura uma execução do fluxo do framework DyDaSL.

        Args:
            ssl_algorithm: algoritmo semissupervisionado que será
                utilizado no comitê
            base_classifiers: lista dos classificadores que 
            detector: classe que realiza a detecção de mudanças de
                contexto
            reactor: classe que realiza a reação de mudanças de
                contexto
        """
        self.ensemble = Ensemble(ssl_algorithm)
        self.detection = detector(params_detector)
        self.reaction = reactor(params_reactor)

    def configure_classifier(self, base_classifiers: list):
        """
        Função para configurar o comitê básico

        Args:
            base_classifiers: lista dos classificadores que irão compor
                o comitê da primeira iteração
        """
        if not self._base_classifiers:
            self._base_classifiers = base_classifiers
        if self.ensemble:
            for classifier in base_classifiers:
                self.ensemble.add_classifier(classifier)
        else:
            raise ValueError(
                "variável 'ensemble' não foi instanciada. "
                "Utilize a função 'configure_params' para preparar o ambiente."
                )

    def run(self, chunk):
        run_first_it()

        y_pred = self.ensemble.predict(chunk)

        if self.detection.detect():
            self.reaction.react()

        self.log_iteration_info()

        pass

    def plot_graph(self, metric, color):
        pass

    def log_iteration_info(self):
        ...
