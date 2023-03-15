from numpy import argmin, ndarray
from skmultiflow.trees import HoeffdingTreeClassifier as HT

from src.reaction.interfaces.reactor import Reactor
from src.ssl.ensemble import Ensemble


class Exchange(Reactor):
    """Módulo de reação ao drift por troca de classificador"""
    def __init__(self, **params):
        self.classifier = params.get("classifier", HT)
        self.retrain_classifier = params.get("retrain_classifier", True)

    def react(
        self,
        ensemble: Ensemble,
        instances: ndarray,
        labels: ndarray,
    ) -> Ensemble:
        y_pred_classifier = ensemble.measure_ensemble(instances, labels)
        ensemble.swap(
            [self.classifier()],
            [argmin(y_pred_classifier)],
            instances,
            labels,
            self.retrain_classifier
        )

        return ensemble
