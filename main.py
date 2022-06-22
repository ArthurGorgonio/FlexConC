from pprint import pprint
import warnings

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB as Naive
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as Tree

from src.ssl.ensemble import Ensemble
from src.ssl.self_flexcon import SelfFlexCon
from src.detection.weighted_statistical import WeightedStatistical

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
#     f"ACC: {round(accuracy_score(y_true, y_pred), 4)}%\n"
#     f'F1-Score: {round(f1_score(y_true, y_pred, average="macro"), 4)}%\n'
#     f"Motivo da finalização: {ssl.termination_condition_}"
# )

comite = Ensemble(SelfFlexCon)
comite.add_classifier(Naive())
comite.add_classifier(Naive(var_smoothing=1e-8))
comite.add_classifier(Naive(var_smoothing=1e7))
comite.add_classifier(Naive(var_smoothing=1e6))
comite.add_classifier(Naive(var_smoothing=1e5))
comite.add_classifier(Naive(var_smoothing=1e4))
import ipdb
ipdb.sset_trace()
from river.datasets import synth
from river import metrics
from river import tree
from river import evaluate


gen = synth.Agrawal(classification_function=0, seed=42)
model = tree.HoeffdingTreeClassifier(
    grace_period=100,
    delta=1e-5,
    nominal_attributes=['elevel', 'car', 'zipcode']
)
dataset = iter(gen.take(5000))

metric = metrics.ClassificationReport()
evaluate.progressive_val_score(dataset, model, metric)
print(metric)


# comite.add_classifier(Tree(criterion="entropy"))
# comite.add_classifier(KNN())

# for x, y in dataset:
#     pprint(x)
#     print(y)
#     break

# rng = np.random.RandomState(42)
# iris = datasets.load_iris()
# random_unlabeled_points = rng.rand(iris.target.shape[0]) < 0.9
# iris.target_unlabelled = iris.target.copy()
# iris.target_unlabelled[random_unlabeled_points] = -1

# comite.fit_ensemble(iris.data, iris.target_unlabelled)

# y_pred = comite.predict(iris.data[random_unlabeled_points, :])
# y_true = iris.target[random_unlabeled_points]

# s_test = WeightedStatistical(0.05)
# s_test.update_chunks(y_pred)
# s_test.update_chunks(y_true)
# alpha = s_test.eval_test()

# print(
#     f"ACC: {round(accuracy_score(y_true, y_pred), 4)}%\n"
#     f'F1-Score: {round(f1_score(y_true, y_pred, average="macro"), 4)}%\n'
#     f"Motivo da finalização: {comite.ensemble[0].termination_condition_}\n"
#     f"Valor do teste estatístico é de {alpha}, significante? {alpha <= 0.05}"
# )
