from typing import Callable

from skmultiflow.trees import HoeffdingTreeClassifier as HT

from src.reaction.interfaces.IReaction import IReaction
from src.ssl.ensemble import Ensemble


class Exchange(IReaction):
    def __init__(self, classifier: Callable = HT, thr: float = 0.8):
        self.classifier = classifier()
        self.thr = thr

    def react(self, ensemble: Ensemble, instances, labels) -> Ensemble:
        y_pred_classifier = ensemble.measure_classifier(instances, labels)
        pos = [p for p, acc in enumerate(y_pred_classifier) if acc < self.thr]
        ensemble.swap(self.classifier, pos)
        return ensemble

