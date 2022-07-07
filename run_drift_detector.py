from typing import List
import numpy as np

# import scipy as sp
import pandas as pd
from sklearn.metrics import accuracy_score
from skmultiflow.bayes import NaiveBayes
from skmultiflow.data import DataStream
from skmultiflow.trees import HoeffdingTreeClassifier
from src.ssl.ensemble import Ensemble
from src.ssl.self_flexcon import SelfFlexCon

## não mutável
class DriftDetectorThreshold:
    _counter = 0

    def __init__(self) -> None:
        self.thr1 = 0.0
        self.thr2 = 0.0

    def react(self):
        if self._counter > 10:
            return True

        return False

    def detect(self, value):
        if value <= 0.8:
            self._counter += 1

            return True

        return False

    def detect_better(self, value, thr: bool = False):
        if value < self.thr1 and thr:
            return True
        elif value < self.thr2 and not thr:
            return True

        return False


class DriftDetectorEnsemble(DriftDetectorThreshold):
    def __init__(self):
        self.all_thrs: List[float] = []
        self.thr_when_drift: List[float] = []
        self.thr_old = 0.0
        self.thr_new = 0.8

    def need_to_retrain(self, ensemble, instances, labels, thr: float = 0.8):
        return (
            np.array(ensemble.measure_classifier(instances, labels)) < thr
        ).tolist()

    def react_mean_all(self, ensemble, instances, labels):
        r_model = self.need_to_retrain(
            ensemble, instances, labels, np.mean(self.all_thrs)
        )

        for i, retrain in enumerate(r_model):
            if retrain:
                ensemble.ensemble[i].partial_fit(instances, labels)

        return ensemble

    def react_mean_drift(self, ensemble, instances, labels):
        r_model = self.need_to_retrain(
            ensemble, instances, labels, np.mean(self.thr_when_drift)
        )

        for i, retrain in enumerate(r_model):
            if retrain:
                ensemble.ensemble[i].partial_fit(instances, labels)

        return ensemble

    def react_retain_when_drop(self, ensemble, instances, labels):
        r_model = self.need_to_retrain(
            ensemble, instances, labels, max(self.thr_old, self.thr_new)
        )

        for i, retrain in enumerate(r_model):
            if retrain:
                ensemble.ensemble[i].partial_fit(instances, labels)

        return ensemble

    def react_varable_thr(self, ensemble, instances, labels):
        r_model = self.need_to_retrain(ensemble, instances, labels, self.thr1)

        for i, retrain in enumerate(r_model):
            if retrain:
                ensemble.ensemble[i].partial_fit(instances, labels)

        return ensemble

    def detect(self, ensemble, instances, labels):
        retrain_models = self.need_to_retrain(ensemble, instances, labels)

        for i, retrain in enumerate(retrain_models):
            if retrain:
                ensemble.ensemble[i].partial_fit(instances, labels)

        return ensemble


num = 1

chunk_size = 500
labelled_per_chunk = 0.1
ensemble1 = Ensemble(SelfFlexCon)
ensemble2 = Ensemble(SelfFlexCon)
ensemble3 = Ensemble(SelfFlexCon)
ensemble4 = Ensemble(SelfFlexCon)
ensemble5 = Ensemble(SelfFlexCon)

# datasets = ["electricity.csv", "ForestCover.csv", "gears2c2d.csv"]
datasets = ["electricity.csv", "ForestCover.csv"]

for dataset in datasets:
    dataframe = pd.read_csv(dataset)
    # depende do dataset
    dim = dataframe.shape
    array = dataframe.values
    instances = array[: dim[0], : dim[1] - 1]
    target = array[: dim[0], dim[1] - 1]
    class_set = np.unique(target)
    class_count = np.unique(target).shape[0]
    stream = DataStream(
        instances,
        target,
        target_idx=-1,
        n_targets=class_count,
        cat_features=None,  # Categorical features?
        name=None,
        allow_nan=True,
    )

    # classifier1 = HoeffdingTreeClassifier()  # detect and retrain
    # classifier2 = HoeffdingTreeClassifier()  # detect and drop
    # base1 = HoeffdingTreeClassifier()  # Train all time
    # base2 = HoeffdingTreeClassifier()  # NO train

    instances_chunk, target_chunk = stream.next_sample(chunk_size)
    num_samples = instances.shape[0] - instances_chunk.shape[0]

    ensemble1.add_classifier(NaiveBayes(), False)
    ensemble1.add_classifier(HoeffdingTreeClassifier(), False)

    ensemble2.add_classifier(NaiveBayes(), False)
    ensemble2.add_classifier(HoeffdingTreeClassifier(), False)

    ensemble3.add_classifier(NaiveBayes(), False)
    ensemble3.add_classifier(HoeffdingTreeClassifier(), False)

    ensemble4.add_classifier(NaiveBayes(), False)
    ensemble4.add_classifier(HoeffdingTreeClassifier(), False)

    ensemble5.add_classifier(NaiveBayes(), False)
    ensemble5.add_classifier(HoeffdingTreeClassifier(), False)

    # classifier1.fit(instances_chunk, target_chunk)
    # classifier2.fit(instances_chunk, target_chunk)
    # base1.fit(instances_chunk, target_chunk)
    # base2.fit(instances_chunk, target_chunk)
    ensemble1.fit_ensemble(instances_chunk, target_chunk)
    ensemble2.fit_ensemble(instances_chunk, target_chunk)
    ensemble3.fit_ensemble(instances_chunk, target_chunk)
    ensemble4.fit_ensemble(instances_chunk, target_chunk)
    ensemble5.fit_ensemble(instances_chunk, target_chunk)

    drift = DriftDetectorThreshold()
    drift_ensemble = DriftDetectorEnsemble()
    acc_ensemble1 = []
    acc_ensemble2 = []
    acc_ensemble3 = []
    acc_ensemble4 = []
    acc_ensemble5 = []

    it1, it2, it3, it4, it5 = True, True, True, True, True

    while stream.has_more_samples():
        instances_chunk_n, target_chunk_n = stream.next_sample(chunk_size)
        #        y_pred1 = classifier1.predict(instances_chunk_n)
        #        y_pred2 = classifier2.predict(instances_chunk_n)
        #        y_pred_base1 = base1.predict(instances_chunk_n)
        #        y_pred_base2 = base2.predict(instances_chunk_n)
        y_pred_ensemble1 = ensemble1.predict(instances_chunk_n)
        y_pred_ensemble2 = ensemble2.predict(instances_chunk_n)
        y_pred_ensemble3 = ensemble3.predict(instances_chunk_n)
        y_pred_ensemble4 = ensemble4.predict(instances_chunk_n)
        y_pred_ensemble5 = ensemble5.predict(instances_chunk_n)

        #        acc_scor1.append(accuracy_score(target_chunk_n, y_pred1))
        #        acc_scor2.append(accuracy_score(target_chunk_n, y_pred2))
        #        acc_base1.append(accuracy_score(target_chunk_n, y_pred_base1))
        #        acc_base2.append(accuracy_score(target_chunk_n, y_pred_base2))
        acc_ensemble1.append(accuracy_score(target_chunk_n, y_pred_ensemble1))
        acc_ensemble2.append(accuracy_score(target_chunk_n, y_pred_ensemble2))
        acc_ensemble3.append(accuracy_score(target_chunk_n, y_pred_ensemble3))
        acc_ensemble4.append(accuracy_score(target_chunk_n, y_pred_ensemble4))
        acc_ensemble5.append(accuracy_score(target_chunk_n, y_pred_ensemble5))

        #         if it1:
        #             drift.thr1 = acc_scor1[-1]
        #             it1 = False
        #
        #         if it2:
        #             drift.thr2 = acc_scor2[-1]
        #             it2 = False

        if it1:
            it1 = False

        if it2:
            drift_ensemble.thr1 = acc_ensemble2[-1]

        if it3:
            ...

        if it4:
            drift_ensemble.thr_when_drift.append(acc_ensemble4[-1])

        if it5:
            drift_ensemble.thr_old = (
                drift_ensemble.thr_new
                if drift_ensemble.thr_new < acc_ensemble5[-1]
                else drift_ensemble.thr_old
            )
            drift_ensemble.thr2 = max(
                drift_ensemble.thr_new, drift_ensemble.thr_old
            )
        #        if drift.detect_better(acc_scor1[-1], True):
        #            classifier1.partial_fit(instances_chunk_n, target_chunk_n)
        #            it1 = True
        #
        #        if drift.detect_better(acc_scor2[-1], False):
        #            classifier2 = clone(_clone)
        #            classifier2.partial_fit(instances_chunk_n, target_chunk_n)
        #            it2 = True

        # DyDaSL - FT

        if acc_ensemble1[-1] < 0.8:
            ensemble1 = drift_ensemble.detect(
                ensemble1, instances_chunk_n, target_chunk_n
            )
            it1 = True

        # DyDaSL - N - Usa o último thr para ver o drift

        if acc_ensemble2[-1] < drift_ensemble.thr1:
            ensemble2 = drift_ensemble.react_varable_thr(
                ensemble2, instances_chunk_n, target_chunk_n
            )
            it2 = True

        # DyDaSL - N - Usa a média de todos os thr até o momento

        if acc_ensemble3[-1] < np.mean(acc_ensemble3):
            ensemble3 = drift_ensemble.react_mean_all(
                ensemble3, instances_chunk_n, target_chunk_n
            )
            it3 = True

        # DyDaSL - N - Usa a média das acc quando foi identificado drift

        if acc_ensemble4[-1] < np.mean(drift_ensemble.thr_when_drift):
            ensemble4 = drift_ensemble.react_mean_drift(
                ensemble4, instances_chunk_n, target_chunk_n
            )
            it4 = True

        # DyDaSL - N

        if acc_ensemble5[-1] < drift_ensemble.thr2:
            ensemble5 = drift_ensemble.react_retain_when_drop(
                ensemble5, instances_chunk_n, target_chunk_n
            )
            it5 = True

    #        base1.partial_fit(instances_chunk_n, target_chunk_n)
    # print(f'ACC:  {acc_scor1[-1]:2f}')

    # import ipdb

    # ipdb.sset_trace()
    #    cum_mean = [
    #        round(i, 4)
    #        for i in (
    #            np.cumsum(acc_scor1) / np.arange(1, len(acc_scor1) + 1)
    #        ).tolist()
    #    ]
    #    print(
    #        f"Media = {np.mean(np.array(acc_scor1))}\t"
    #        f"Acumulada = {cum_mean[-1]}\n\n"
    #    )

    data = {
        "chunk": np.arange(1, len(acc_ensemble1) + 1),
        #        "mean_train_retrain": acc_scor1,
        #        "mean_train_drop": acc_scor2,
        #        "mean_train_all": acc_base1,
        #        "mean_no_train": acc_base2,
        "mean_ensemble1": acc_ensemble1,
        "mean_ensemble2": acc_ensemble2,
        "mean_ensemble3": acc_ensemble3,
        "mean_ensemble4": acc_ensemble4,
        "mean_ensemble5": acc_ensemble5,
        #        "cummean_train_retrain": [
        #            round(i, 4)
        #            for i in (
        #                np.cumsum(acc_scor1) / np.arange(1, len(acc_scor1) + 1)
        #            ).tolist()
        #        ],
        #        "cummean_train_drop": [
        #            round(i, 4)
        #            for i in (
        #                np.cumsum(acc_scor2) / np.arange(1, len(acc_scor1) + 1)
        #            ).tolist()
        #        ],
        #        "cummean_train_all": [
        #            round(i, 4)
        #            for i in (
        #                np.cumsum(acc_base1) / np.arange(1, len(acc_scor1) + 1)
        #            ).tolist()
        #        ],
        #        "cummean_no_train": [
        #            round(i, 4)
        #            for i in (
        #                np.cumsum(acc_base2) / np.arange(1, len(acc_scor1) + 1)
        #            ).tolist()
        #        ],
        "cummean_ensemble1": [
            round(i, 4)
            for i in (
                np.cumsum(acc_ensemble1) / np.arange(1, len(acc_ensemble1) + 1)
            ).tolist()
        ],
        "cummean_ensemble2": [
            round(i, 4)
            for i in (
                np.cumsum(acc_ensemble2) / np.arange(1, len(acc_ensemble2) + 1)
            ).tolist()
        ],
        "cummean_ensemble3": [
            round(i, 4)
            for i in (
                np.cumsum(acc_ensemble3) / np.arange(1, len(acc_ensemble3) + 1)
            ).tolist()
        ],
        "cummean_ensemble4": [
            round(i, 4)
            for i in (
                np.cumsum(acc_ensemble4) / np.arange(1, len(acc_ensemble4) + 1)
            ).tolist()
        ],
        "cummean_ensemble5": [
            round(i, 4)
            for i in (
                np.cumsum(acc_ensemble5) / np.arange(1, len(acc_ensemble5) + 1)
            ).tolist()
        ],
    }

    pd.DataFrame(data).to_csv(
        f"{dataset.split('.csv')[0]}_five_ensembles.dat",
        sep="\t",
        header=False,
        index=False,
    )
