from time import time
from typing import Callable, Dict

from numpy import ndarray
from sklearn.metrics import accuracy_score, confusion_matrix
from skmultiflow.data import DataStream
from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier as HT

from src.detection.fixed_threshold import FixedThreshold
from src.detection.interfaces.drift_detector import DriftDetector
from src.reaction.exchange import Exchange
from src.reaction.interfaces.ireaction import IReaction
from src.ssl.ensemble import Ensemble
from src.utils import Log


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
        detector: DriftDetector = None,
        reactor: IReaction = None,
        chunk_size: int = 500
    ):
        if ensemble is None:
            ensemble = Ensemble
        if detector is None:
            detector = FixedThreshold
        if reactor is None:
            reactor = Exchange
        self.ensemble = ensemble
        self.detector = detector
        self.reactor = reactor
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
        self.detector = self.detector(**params_detector or {})
        self.reactor = self.reactor(**params_reactor or {})

    def configure_classifier(self, base_classifiers: list) -> None:
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

        while chunk.has_more_samples():
            start = time()
            instances, classes = chunk.next_sample(self.chunk_size)
            y_pred = self.ensemble.predict_ensemble(instances)
            drift = self._detect_by_type(
                instances,
                classes,
                y_pred,
                self.detector.detector_type
            )
            if drift:
                self.reactor.react(
                    self.ensemble,
                    instances,
                    classes,
                    self.detector.detection_threshold
                )
            enlapsed_time = time() - start
            self._evaluate_metrics(classes, y_pred)
            hits = confusion_matrix(classes, y_pred)
            self._log_iteration_info(
                sum(hits.diagonal()),
                chunk.sample_idx,
                enlapsed_time
            )
            if len(self.ensemble.ensemble) < 10:
                self.run_first_it(instances, classes)

        return self

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
        self.ensemble.fit_single_classifier(
            self.ensemble.ensemble[-1],
            instances,
            classes
        )

        return self

    def _detect_by_type(
        self,
        instances: ndarray,
        classes: ndarray,
        y_pred: ndarray,
        is_metric_value: str,
    ) -> bool:
        if is_metric_value == 'metric':
            thr = accuracy_score(classes, y_pred)
            return self.detector.detect(thr)
        if is_metric_value == 'classes':
            return self.detector.detect(classes)
        return self.detector.detect(instances)

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

    def _evaluate_metrics(self, y_true: ndarray, y_pred: ndarray):
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
            if func_name == 'f1':
                self.metrics[func_name].append(
                        metric(y_true, y_pred, average='macro'))
            else:
                self.metrics[func_name].append(metric(y_true, y_pred))

    def _log_iteration_info(self, hits, processed, enlapsed_time):
        # version = self.detector.__class__
        iteration_info = {
            'ensemble_size': len(self.ensemble.ensemble),
            'ensemble_hits': hits,
            'drift_detected': self.detector.drift,
            'instances': processed,
            'enlapsed_time': enlapsed_time,
            'metrics': {
                'acc': self.metrics['acc'][-1],
                'f1': self.metrics['f1'][-1],
                'kappa': self.metrics['kappa'][-1],
            },
        }
        Log().write_archive_output(**iteration_info)

    def reset(self):
        self.detector.reset_params()
        self.ensemble.drop_ensemble()
