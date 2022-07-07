import warnings

import src.utils as preprocessing
import pandas as pd
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
# digits = datasets.load_digits()

#digits_target_unlabelled = digits.target.copy()

input_file = "pendigits.tra"

df = pd.read_csv(input_file, header=None)

kfold = preprocessing.crossValidation(10)
digits_instances = df.iloc[:,:-1].values #X
digits_target_unlabelled = df.iloc[:,-1].values #Y
digits_target_unlabelled_copy = digits_target_unlabelled.copy()

# TODO: DPS
# for train, test in kfold.split(digits_instances, digits_target_unlabelled):
#     X_train, X_test = digits_instances[train], digits_instances[test]
#     y_train, y_test = digits_target_unlabelled[train], digits_target_unlabelled[test]


# flexCon = SelfFlexCon(Tree())

list_tree_het = [
    Tree(criterion="entropy"), Tree(), 
    Tree(criterion="entropy", max_features="log2"),
    Tree(criterion="entropy", max_features='auto'), Tree(max_features='auto')]

list_knn_het = [
    KNN(n_neighbors=11),KNN(n_neighbors=12),
    KNN(n_neighbors=13),KNN(n_neighbors=14),
    KNN(n_neighbors=15)]

list_tree = [
    Tree(criterion="entropy"), Tree(), 
    Tree(criterion="entropy", max_features=1), Tree(max_features=1),
    Tree(criterion="entropy", max_features="log2"),
    Tree(criterion="entropy", max_features='auto'), Tree(max_features='auto'),
    Tree(splitter="random"), Tree(criterion="entropy",splitter="random")]

list_knn= [
    KNN(n_neighbors=10, weights='distance'),KNN(n_neighbors=11, weights='distance'),
    KNN(n_neighbors=12, weights='distance'),KNN(n_neighbors=13, weights='distance'),
    KNN(n_neighbors=14, weights='distance'),KNN(n_neighbors=11),KNN(n_neighbors=12),
    KNN(n_neighbors=13),KNN(n_neighbors=14)]

labelled_instances = int(len(digits_instances)*0.2)
print('Instâncias rotuladas: ', labelled_instances)
print('Data-Set carregado: Digits\n\n')
print("Classificadores disponíveis:\n\n\t(1) Naive\n\t(2) Tree\n\t(3) KNN\n\t(4) COMITE HETEROGENEO\n\t(0) Sair\n\n")
option = input('Informe o classificador: ')
print('\nExecutando o treinamento...\n\n')
flag = False
while flag == False:
    if option == 'Sair' or option == '0':
        print('\nEncerrando...\n')
        exit()
    elif option == 'Naive' or option == '1':
        flag = True
        with open('Comite_Naive.txt', 'a') as f:
            f.write('Naive Bayes selecionado...\n\n')
        for i in range(10):
            flexCon = SelfFlexCon(Naive(var_smoothing=float(f'1e{i}')))
            random_unlabeled_points = np.random.choice(len(digits_instances), labelled_instances, replace=False)
            digits_target_unlabelled[random_unlabeled_points] = -1
            X = digits_instances
            y = digits_target_unlabelled
            comite.add_model(flexCon.fit(X, y, option))

    elif option == 'Tree' or option == '2':
        flag = True
        with open('Comite_Tree.txt', 'a') as f:
            f.write('Decision Tree selecionado...\n\n')
        for i in list_tree:
            flexCon = SelfFlexCon(i)
            random_unlabeled_points = np.random.choice(len(digits_instances), labelled_instances, replace=False)
            digits_target_unlabelled[random_unlabeled_points] = -1
            X = digits_instances
            y = digits_target_unlabelled
            comite.add_model(flexCon.fit(X, y, option))

    elif option == 'KNN' or option == '3':
        flag = True
        with open('Comite_KNN.txt', 'a') as f:
            f.write('\n\nKNN selecionado...\n\n')
        for i in list_knn:
            flexCon = SelfFlexCon(i)
            random_unlabeled_points = np.random.choice(len(digits_instances), labelled_instances, replace=False)
            digits_target_unlabelled[random_unlabeled_points] = -1
            X = digits_instances
            y = digits_target_unlabelled
            comite.add_model(flexCon.fit(X, y, option))

    elif option == 'COMITE HETEROGENEO' or option == '4':
        flag = True
        with open('Comite_Heterogeneo.txt', 'a') as f:
            f.write('Comite heterogêneo selecionado...\n\n'
                    'Executando com Naive Bayes...\n\n')
        for i in range(5):
            flexCon = SelfFlexCon(Naive(var_smoothing=float(f'1e{i}')))
            random_unlabeled_points = np.random.choice(len(digits_instances), labelled_instances, replace=False)
            digits_target_unlabelled[random_unlabeled_points] = -1
            X = digits_instances 
            y = digits_target_unlabelled
            comite.add_model(flexCon.fit(X, y, option))
        with open('Comite_Heterogeneo.txt', 'a') as f:
            f.write('\n\nExecutando com Decision Tree...\n\n')
        for i in list_tree_het:
            flexCon = SelfFlexCon(i)
            random_unlabeled_points = np.random.choice(len(digits_instances), labelled_instances, replace=False)
            digits_target_unlabelled[random_unlabeled_points] = -1
            X = digits_instances
            y = digits_target_unlabelled
            comite.add_model(flexCon.fit(X, y, option))
        with open('Comite_Heterogeneo.txt', 'a') as f:
            f.write('\n\nExecutando com KNN...\n\n')
        for i in list_knn_het:
            flexCon = SelfFlexCon(i)
            random_unlabeled_points = np.random.choice(len(digits_instances), labelled_instances, replace=False)
            digits_target_unlabelled[random_unlabeled_points] = -1
            X = digits_instances
            y = digits_target_unlabelled
            comite.add_model(flexCon.fit(X, y, option))

    else:
        print('Classificador não disponível! Insira outro...\n')
        print("Classificadores disponíveis:\n\n\t(1) Naive\n\t(2) Tree\n\t(3) KNN\n\t(4) COMITE HETEROGENEO\n\t(0) Sair\n\n")
        option = input('Informe o classificador: ')


# comite.add_classifier(Tree(criterion="entropy"))
# comite.add_classifier(KNN())

# comite.fit_ensemble(digits.data,digits_target_unlabelled)

y_pred = comite.predict(digits_instances[random_unlabeled_points, :])
y_true = digits_target_unlabelled_copy[random_unlabeled_points]

s_test = WeightedStatistical(0.05)
s_test.update_chunks(y_pred)
s_test.update_chunks(y_true)
alpha = s_test.eval_test()

if option == 'Naive' or option == '1':
    print('Salvando os resultados em um arquivo Comite_Naive.txt\n\n')
    print('Finalizando...')
    with open('Comite_Naive.txt', 'a') as f:
        f.write(
            f"\n\nACC: {round(accuracy_score(y_true, y_pred), 4) * 100}%\n"
            f'F1-Score: {round(f1_score(y_true, y_pred, average="macro"), 4) * 100}%\n'
            f"Motivo da finalização: {comite.ensemble[0].termination_condition_}\n"
            # f"Valor do teste estatístico é de {alpha}, significante? {alpha <= 0.05}\n"
        )

elif option == 'Tree' or option == '2':
    print('Salvando os resultados em um arquivo Comite_Tree.txt\n\n')
    print('Finalizando...')
    with open('Comite_Tree.txt', 'a') as f:
        f.write(
            f"\n\nACC: {round(accuracy_score(y_true, y_pred), 4) * 100}%\n"
            f'F1-Score: {round(f1_score(y_true, y_pred, average="macro"), 4) * 100}%\n'
            f"Motivo da finalização: {comite.ensemble[0].termination_condition_}\n"
            # f"Valor do teste estatístico é de {alpha}, significante? {alpha <= 0.05}\n"
        )

elif option == 'KNN' or option == '3':
    print('Salvando os resultados em um arquivo Comite_KNN.txt\n\n')
    print('Finalizando...')
    with open('Comite_KNN.txt', 'a') as f:
        f.write(
            f"\n\nACC: {round(accuracy_score(y_true, y_pred), 4) * 100}%\n"
            f'F1-Score: {round(f1_score(y_true, y_pred, average="macro"), 4) * 100}%\n'
            f"Motivo da finalização: {comite.ensemble[0].termination_condition_}\n"
            # f"Valor do teste estatístico é de {alpha}, significante? {alpha <= 0.05}\n"
        )
elif option == 'COMITE HETEROGENEO' or option == '4':
    print('Salvando os resultados em um arquivo Comite_Heterogeneo.txt\n\n')
    print('Finalizando...')
    with open('Comite_Heterogeneo.txt', 'a') as f:
        f.write(
            f"\n\nACC: {round(accuracy_score(y_true, y_pred), 4) * 100}%\n"
            f'F1-Score: {round(f1_score(y_true, y_pred, average="macro"), 4) * 100}%\n'
            f"Motivo da finalização: {comite.ensemble[0].termination_condition_}\n"
            # f"Valor do teste estatístico é de {alpha}, significante? {alpha <= 0.05}\n"
        )
