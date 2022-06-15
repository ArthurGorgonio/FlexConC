import warnings

import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB as Naive
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as Tree

from src.detection.weighted_statistical import WeightedStatistical
from src.ssl.ensemble import Ensemble
from src.ssl.self_flexcon import SelfFlexCon

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

# iris = datasets.load_iris()
# wine = datasets.load_wine()
digits = datasets.load_digits()

digits.target_unlabelled = digits.target.copy()

# flexCon = SelfFlexCon(Tree())

list_tree = [
    Tree(criterion="entropy"), Tree(), 
    Tree(criterion="entropy", max_features=1), Tree(max_features=1),
    Tree(criterion="entropy", max_features="log2"),Tree(max_features="log2"),
    Tree(criterion="entropy", max_features='auto'), Tree(max_features='auto'), 
    Tree(splitter="random"), Tree(criterion="entropy",splitter="random")]

list_knn = [
    KNN(n_neighbors=10, weights='distance'),KNN(n_neighbors=11, weights='distance'),
    KNN(n_neighbors=12, weights='distance'),KNN(n_neighbors=13, weights='distance'),
    KNN(n_neighbors=14, weights='distance'),KNN(n_neighbors=10),
    KNN(n_neighbors=11),KNN(n_neighbors=12),
    KNN(n_neighbors=13),KNN(n_neighbors=14)
]
# print('Data-Set tamanho: ', len(digits.data))
print('Data-Set carregado: Digits\n\n')
print("Classificadores disponíveis:\n\n\t(1) Naive\n\t(2) Tree\n\t(3) KNN\n\t(0) Sair\n\n")
option = input('Informe o classificador: ')
flag = False
while flag == False:
    if option == 'Sair' or option == '0':
        print('\nEncerrando...\n')
        exit()
    elif option == 'Naive' or option == '1':
        flag = True
        print('Naive Bayes selecionado...\n\n')
        for i in range(10):
            flexCon = SelfFlexCon(Naive(var_smoothing=float(f'1e{i}')))
            random_unlabeled_points = np.random.choice(len(digits.data), 15, replace=False)
            digits.target_unlabelled[random_unlabeled_points] = -1
            X = digits.data
            y = digits.target_unlabelled
            comite.add_model(flexCon.fit(X, y))

    elif option == 'Tree' or option == '2':
        flag = True
        print('Decision Tree selecionado...\n\n')
        for i in list_tree:
            flexCon = SelfFlexCon(i)
            random_unlabeled_points = np.random.choice(len(digits.data), 15, replace=False)
            digits.target_unlabelled[random_unlabeled_points] = -1
            X = digits.data
            y = digits.target_unlabelled
            comite.add_model(flexCon.fit(X, y))

    elif option == 'KNN' or option == '3':
        flag = True
        print('K-NN selecionado...\n\n')
        for i in list_knn:
            flexCon = SelfFlexCon(i)
            random_unlabeled_points = np.random.choice(len(digits.data), 15, replace=False)
            digits.target_unlabelled[random_unlabeled_points] = -1
            X = digits.data
            y = digits.target_unlabelled
            comite.add_model(flexCon.fit(X, y))
    else:
        print('Classificador não disponível! Insira outro...\n')
        print("Classificadores disponíveis:\n\n\t(1) Naive\n\t(2) Tree\n\t(3) KNN\n\t(0) Sair\n\n")
        option = input('Informe o classificador: ')


# comite.add_classifier(Tree(criterion="entropy"))
# comite.add_classifier(KNN())

# comite.fit_ensemble(digits.data, digits.target_unlabelled)

y_pred = comite.predict(digits.data[random_unlabeled_points, :])
y_true = digits.target[random_unlabeled_points]

s_test = WeightedStatistical(0.05)
s_test.update_chunks(y_pred)
s_test.update_chunks(y_true)
alpha = s_test.eval_test()

print(
    f"ACC: {round(accuracy_score(y_true, y_pred), 4)}%\n"
    f'F1-Score: {round(f1_score(y_true, y_pred, average="macro"), 4)}%\n'
    f"Motivo da finalização: {comite.ensemble[0].termination_condition_}\n"
    f"Valor do teste estatístico é de {alpha}, significante? {alpha <= 0.05}\n"
)
