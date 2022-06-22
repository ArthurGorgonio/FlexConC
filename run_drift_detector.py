import numpy as np
import scipy as sp
from pandas import read_csv
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
    target_idx=0,
    n_targets=class_count,
    cat_features=None,
    name=None,
    allow_nan=False,
)


print(
    f'{len(class_set)}\n'
    f'{class_set}\n\n'
    f'{class_count}'
)