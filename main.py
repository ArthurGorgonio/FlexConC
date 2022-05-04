import warnings

import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB as Naive

from src.ensemble import Ensemble

warnings.simplefilter("ignore")
# ssl = FlexConC(Naive(), verbose=True)

# rng = np.random.RandomState(42)
# iris = datasets.load_iris()
# random_unlabeled_points = rng.rand(iris.target.shape[0]) < 0.9
# iris.target_unlabelled = iris.target.copy()
# iris.target_unlabelled[random_unlabeled_points] = -1

# ssl.fit(iris.data, iris.target_unlabelled)

# y_pred = ssl.predict(iris.data[random_unlabeled_points, :])
# y_true = iris.target[random_unlabeled_points]

# print(
#     f"ACC: {round(accuracy_score(y_true, y_pred), 2)}%\n"
#     f'F1-Score: {round(f1_score(y_true, y_pred, average="macro"), 2)}%\n'
#     f"Motivo da finalização: {ssl.termination_condition_}"
# )

comite = Ensemble()
comite.add_classifier(Naive())


rng = np.random.RandomState(42)
iris = datasets.load_iris()
random_unlabeled_points = rng.rand(iris.target.shape[0]) < 0.9
iris.target_unlabelled = iris.target.copy()
iris.target_unlabelled[random_unlabeled_points] = -1

comite.fit_ensembĺe(iris.data, iris.target_unlabelled)

y_pred = comite.predict(iris.data[random_unlabeled_points, :])
y_true = iris.target[random_unlabeled_points]

print(
    f"ACC: {round(accuracy_score(y_true, y_pred), 2)}%\n"
    f'F1-Score: {round(f1_score(y_true, y_pred, average="macro"), 2)}%\n'
    f"Motivo da finalização: {comite.ensemble[0].termination_condition_}"
)
