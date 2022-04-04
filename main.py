import numpy as np
from sklearn import datasets
from src.flexcon import FlexConC

from sklearn.naive_bayes import GaussianNB as Naive
from sklearn.metrics import accuracy_score, f1_score
ssl = FlexConC(Naive(), verbose=True)
print(ssl)

rng = np.random.RandomState(42)
iris = datasets.load_iris()
random_unlabeled_points = rng.rand(iris.target.shape[0]) < 0.9
iris.target_unlabelled = iris.target.copy()
iris.target_unlabelled[random_unlabeled_points] = -1
# 
# model = Naive()
#

ssl.fit(iris.data, iris.target_unlabelled)

y_pred = ssl.predict(iris.data[random_unlabeled_points, :])

print(f'ACC: {round(accuracy_score(iris.target[random_unlabeled_points], y_pred), 2)}%\n'
      f'F1-Score: {round(f1_score(iris.target[random_unlabeled_points], y_pred, average="macro"), 2)}%\n'
      f'Motivo da finalização: {ssl.termination_condition_}')