from typing import List

from sklearn.base import clone
from skmultiflow.tree import HoeffdingTreeClassifier as HT

from src.reaction.interfaces.IReaction import IReaction
from src.ssl.ensemble import Ensemble


class Exchange(IReaction):
    def __init__(self, classifier: callable = HT, thr: float = 0.8):
        self.classifier = classifier()
        self.thr = thr

    # TODO:
    # implement
    # require, define the classifier, probrally it will defined in the init
    #   method, since all instace mey be need to use this one
    def swap_ensemble(self, ensemble, instances, labels):
        # assume that the ensemble
        y_pred_classifier = ensemble.measure_classifier(instances, labels)
        pos = [p for p, acc in enumerate(y_pred_classifier) if acc < self.thr]
        ensemble.

    def react(self, ensemble):
