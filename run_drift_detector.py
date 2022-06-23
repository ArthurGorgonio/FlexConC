import numpy as np
import scipy as sp
from pandas import read_csv
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from skmultiflow.bayes import NaiveBayes
from skmultiflow.data import DataStream
from skmultiflow.trees import HoeffdingTreeClassifier

## não mutável
chunk_size = 500
labelled_per_chunk = 0.1

dataframe = read_csv("electricity.csv")
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

    def react(self):
        if self._counter > 10:
            return True
        return False 

    def detect(self, value):
        if value <= 0.8:
            self._counter += 1
            return True
        return False

num = 1

if num:
    classifier = HoeffdingTreeClassifier()
else:
    classifier = NaiveBayes()

_clone = clone(classifier)
instances_chunk, target_chunk = stream.next_sample(chunk_size)
num_samples = instances.shape[0] - instances_chunk.shape[0]

classifier.fit(instances_chunk, target_chunk)

drift = DriftDetectorThreshold()
acc_score = []

while stream.has_more_samples():
    instances_chunk_n, target_chunk_n = stream.next_sample(chunk_size)
    y_pred = classifier.predict(instances_chunk_n)
    acc_score.append(accuracy_score(target_chunk_n, y_pred))

    if drift.detect(acc_score[-1]):
        if drift.react():
            classifier = clone(_clone)
            classifier.fit(instances_chunk_n, target_chunk_n)
        else:
            classifier.partial_fit(instances_chunk_n, target_chunk_n)
    # print(f'ACC:  {acc_score[-1]:2f}')

import ipdb

ipdb.sset_trace()np.cumsum(np.array(acc_score))
print(
    f'Media = {np.mean(np.array(acc_score))}\t'
    f'Acumulada = {np.cumsum(np.array(acc_score))}\n\n'
)
# pd.DataFrame({'chunk'=np.arange(1,len(acc_score)+1), 'cummean'=[round(i, 4) for i in (np.cumsum(acc_score) / np.arange(1,len(acc_score)+1)).tolist()], 'mean'=acc_score})