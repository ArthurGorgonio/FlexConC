from typing import Callable, Dict, NoReturn
from numpy import ndarray

from skmultiflow.data import DataStream
from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier as HT

from src.detection.fixed_threshold import FixedThreshold
from src.detection.interfaces.idrift_detector import IDriftDetector
from src.reaction.exchange import Exchange
from src.reaction.interfaces.ireaction import IReaction
from src.ssl.ensemble import Ensemble


class Core:
    """
    Classe principal do DyDaSL, gerencia todo o fluxo de treinamento,
    detecção do drift e reação do drift

    Parameters
    ----------
    ensemble : Ensemble
        Um comitê de classificadores.
    detect : IDriftDetector
        O detector de drift que determina quando é necessário alterar a
        stream.
    react : IReaction
        O reator do drift.
    chunk_size : int
        Quantidade de instâncias que serão processadas de uma única vez
    """

    def __init__(
        self,
        ensemble: Ensemble = None,
        detect: IDriftDetector = None,
        react: IReaction = None,
        chunk_size: int = 500
    ):
        if ensemble is None:
            ensemble = Ensemble
        if detect is None:
            detect = FixedThreshold
        if react is None:
            react = Exchange
        self.ensemble = ensemble
        self.detect = detect
        self.react = react
        self.chunk_size = chunk_size
        self.metrics_calls = {}
        self.metrics = {}

    def configure_params(
        self,
        ssl_algorithm: Callable,
        params_ssl_algorithm: Dict[str, any] = None,
        params_detector: Dict[str, any] = None,
        params_reactor: Dict[str, any] = None,
    ):
        """
        Função que configura uma execução do fluxo do framework DyDaSL.

        Parameters
        ----------
        ssl_algorithm :
            algoritmo semissupervisionado que será utilizado no comitê.
        detector :
            classe que realiza a detecção de mudanças de contexto.
        reactor :
            classe que realiza a reação de mudanças de contexto.
        params_detector : Dict[str, any], optional
            parâmetros necessários para o módulo de detecção do drift.
            Se None, serão utilizados os valores padrão da classe.
        params_reactor : Dict[str, any], optional
            parâmetros necessários para , por default None
        """
        self.ensemble = self.ensemble(ssl_algorithm, params_ssl_algorithm)
        self.detect = self.detect(**params_detector)
        self.react = self.react(**params_reactor)

    def configure_classifier(self, base_classifiers: list) -> NoReturn:
        """
        Função para configurar o comitê básico

        Parameters
        ----------
        base_classifiers : list
            lista dos classificadores que irão compor o comitê da
            primeira iteração.
        """
        try:
            for classifier in base_classifiers:
                self.ensemble.add_classifier(classifier)
        except NameError as exc:
            raise NameError(
                "variável 'ensemble' não foi instanciada. "
                "Utilize a função 'configure_params' para preparar o ambiente."
            ) from exc

    def run(self, chunk: DataStream):
        """
        Fluxo de execução do DyDaSL, loop para realizar a classificação
        de instâncias, enquanto houver instâncias disponíveis na stream

        Parameters
        ----------
        chunk : DataStream
            stream a ser classificada.
        """
        instances, classes = chunk.next_sample(self.chunk_size)
        self.run_first_it(instances, classes)

        y_pred = self.ensemble.predict(chunk)

        if self.detect.detect():
            self.react.react()

        self.log_iteration_info()


    def run_first_it(self, instances: ndarray, classes: ndarray):
        """
        Executa a primeira iteração do loop, pois é necessários treinar
        todo o comitê de classificadores antes de poder utilizá-lo para
        rotular as instâncias da stream.

        Parameters
        ----------
        instances : ndarray
            Instâncias a serem utilizadas no treinamento.
        classes : ndarray
            Rótulos das instâncias.
        """
        self.ensemble.add_classifier(HT(), need_train=False)
        self.ensemble.fit_ensemble(instances, classes)

        return self

    def add_metrics(self, metric_name: str, metric_func: Callable) -> None:
        """
        Gerencia as métricas que serão computadas durante e evaluação
        da stream.

        Parameters
        ----------
        metric_name : str
            Nome da métrica que será computada.
        metric_func : Callable
            Função para computar a métrica.
        """
        self.metrics_calls[metric_name] = metric_func
        self.metrics[metric_name] = []

    def evaluate_metrics(self, y_true: ndarray, y_pred: ndarray):
        """
        Computa cada uma das métricas adicionadas a partir do valor
        predito pelo comitê.

        Parameters
        ----------
        y_true : ndarray
            Rótulos verdadeiros.
        y_pred : ndarray
            Rótulos preditos pelo comitê.
        """
        for func_name in self.metrics_calls.keys():
            metric = self.metrics_calls.get(func_name)
            self.metrics[func_name].append(metric(y_true, y_pred))

    def log_iteration_info(self, ):
        ...
