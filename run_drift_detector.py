import numpy as np
# import scipy as sp
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from skmultiflow.bayes import NaiveBayes
from skmultiflow.data import DataStream
from skmultiflow.trees import HoeffdingTreeClassifier

## não mutável
chunk_size = 500
labelled_per_chunk = 0.1

dataframe = pd.read_csv("electricity.csv")
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
    allow_nan=False,
)


class DriftDetectorThreshold:
    _counter = 0
    def __init__(self) -> None:
        self.thr1 = 0
        self.thr2 = 0

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


num = 1

if num:
    classifier1 = HoeffdingTreeClassifier()  # detect and retrain
    classifier2 = HoeffdingTreeClassifier()  # detect and drop
    base1 = HoeffdingTreeClassifier()  # Train all time
    base2 = HoeffdingTreeClassifier()  # NO train
else:
    classifier1 = NaiveBayes()

_clone = clone(classifier1)
instances_chunk, target_chunk = stream.next_sample(chunk_size)
num_samples = instances.shape[0] - instances_chunk.shape[0]

classifier1.fit(instances_chunk, target_chunk)
classifier2.fit(instances_chunk, target_chunk)
base1.fit(instances_chunk, target_chunk)
base2.fit(instances_chunk, target_chunk)

drift = DriftDetectorThreshold()
acc_scor1 = []
acc_scor2 = []
acc_base1 = []
acc_base2 = []

it1, it2 = True, True

while stream.has_more_samples():
    instances_chunk_n, target_chunk_n = stream.next_sample(chunk_size)
    y_pred1 = classifier1.predict(instances_chunk_n)
    y_pred2 = classifier2.predict(instances_chunk_n)
    y_pred_base1 = base1.predict(instances_chunk_n)
    y_pred_base2 = base2.predict(instances_chunk_n)

    acc_scor1.append(accuracy_score(target_chunk_n, y_pred1))
    acc_scor2.append(accuracy_score(target_chunk_n, y_pred2))
    acc_base1.append(accuracy_score(target_chunk_n, y_pred_base1))
    acc_base2.append(accuracy_score(target_chunk_n, y_pred_base2))

    if it1:
        drift.thr1 = acc_scor1[-1]
        it1 = False
    if it2:
        drift.thr2 = acc_scor2[-1]
        it2 = False

    if drift.detect_better(acc_scor1[-1], True):
        classifier1.partial_fit(instances_chunk_n, target_chunk_n)
        it1 = True

    if drift.detect_better(acc_scor2[-1], False):
        classifier2 = clone(_clone)
        classifier2.partial_fit(instances_chunk_n, target_chunk_n)
        it2 = True
        # else:
        #     classifier1.partial_fit(instances_chunk_n, target_chunk_n)
    base1.partial_fit(instances_chunk_n, target_chunk_n)
    # print(f'ACC:  {acc_scor1[-1]:2f}')

# import ipdb

# ipdb.sset_trace()
cum_mean = [
    round(i, 4)
    for i in (np.cumsum(acc_scor1) / np.arange(1, len(acc_scor1) + 1)).tolist()
]
print(
    f"Media = {np.mean(np.array(acc_scor1))}\t"
    f"Acumulada = {cum_mean[-1]}\n\n"
)

data = {
    'chunk': np.arange(1,len(acc_scor1)+1),
    'mean_train_retrain': acc_scor1,
    'mean_train_drop': acc_scor2,
    'mean_train_all': acc_base1,
    'mean_no_train': acc_base2,
    'cummean_train_retrain': [round(i, 4) for i in (np.cumsum(acc_scor1) / np.arange(1,len(acc_scor1)+1)).tolist()],
    'cummean_train_drop': [round(i, 4) for i in (np.cumsum(acc_scor2) / np.arange(1,len(acc_scor1)+1)).tolist()],
    'cummean_train_all': [round(i, 4) for i in (np.cumsum(acc_base1) / np.arange(1,len(acc_scor1)+1)).tolist()],
    'cummean_no_train': [round(i, 4) for i in (np.cumsum(acc_base2) / np.arange(1,len(acc_scor1)+1)).tolist()],
}

pd.DataFrame(data).to_csv('data2.dat', sep='\t', header=None, index=None)
