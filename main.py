import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB as Naive
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as Tree

from src.detection.weighted_statistical import WeightedStatistical
from src.ssl.ensemble import Ensemble
from src.ssl.self_flexcon import SelfFlexCon

warnings.simplefilter("ignore")

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

comite = Ensemble(SelfFlexCon)

# iris = datasets.load_iris()
# wine = datasets.load_wine()
# digits = datasets.load_digits()

#digits_target_unlabelled = digits.target.copy()

input_file = "pendigits.tra"

df = pd.read_csv(input_file, header=None)

kfold = StratifiedKFold(n_splits=10)

digits_instances = df.iloc[:,:-1].values #X

digits_target_unlabelled = df.iloc[:,-1].values #Y

digits_target_unlabelled_copy = digits_target_unlabelled.copy()

print("Classificadores disponíveis:\n\n\t(1) Naive\n\t(2) Tree\n\t(3) KNN\n\t(4) Comite Heterogeneo\n\t(0) Sair\n\n")
option = input('Informe o classificador: ')
print('\nExecutando o treinamento...\n\n')

for train, test in kfold.split(digits_instances, digits_target_unlabelled):
    X_train, X_test = digits_instances[train], digits_instances[test]
    y_train, y_test = digits_target_unlabelled[train], digits_target_unlabelled[test]
    labelled_instances = int(len(X_train)*0.15)

    print('Instâncias rotuladas: ', labelled_instances)
    if option == 'Sair' or option == '0':
        print('\nEncerrando...\n')
        exit()
    elif option == 'Naive' or option == '1':
        with open('Comite_Naive.txt', 'a') as f:
            f.write('Naive Bayes selecionado...\n\n')
        for i in range(9):
            flexCon = SelfFlexCon(Naive(var_smoothing=float(f'1e-{i}')))
            random_unlabeled_points = np.random.choice(len(X_train), labelled_instances, replace=False)
            y_train[random_unlabeled_points] = -1
            X = X_train
            y = y_train
            comite.add_model(flexCon.fit(X, y, option))

    elif option == 'Tree' or option == '2':
        with open('Comite_Tree.txt', 'a') as f:
            f.write('Decision Tree selecionado...\n\n')
        for i in list_tree:
            flexCon = SelfFlexCon(i)
            random_unlabeled_points = np.random.choice(len(X_train), labelled_instances, replace=False)
            y_train[random_unlabeled_points] = -1
            X = X_train
            y = y_train
            comite.add_model(flexCon.fit(X, y, option))

    elif option == 'KNN' or option == '3':
        with open('Comite_KNN.txt', 'a') as f:
            f.write('\n\nKNN selecionado...\n\n')
        for i in list_knn:
            flexCon = SelfFlexCon(i)
            random_unlabeled_points = np.random.choice(len(X_train), labelled_instances, replace=False)
            y_train[random_unlabeled_points] = -1
            X = X_train
            y = y_train
            comite.add_model(flexCon.fit(X, y, option))

    elif option == 'Comite Heterogeneo' or option == '4':
        with open('Comite_Heterogeneo.txt', 'a') as f:
            f.write('Comite heterogêneo selecionado...\n\n'
                    'Executando com Naive Bayes...\n\n')
        for i in range(5):
            flexCon = SelfFlexCon(Naive(var_smoothing=float(f'1e-{i}')))
            random_unlabeled_points = np.random.choice(len(X_train), labelled_instances, replace=False)
            y_train[random_unlabeled_points] = -1
            X = X_train 
            y = y_train
            comite.add_model(flexCon.fit(X, y, option))
        with open('Comite_Heterogeneo.txt', 'a') as f:
            f.write('\n\nExecutando com Decision Tree...\n\n')
        for i in list_tree_het:
            flexCon = SelfFlexCon(i)
            random_unlabeled_points = np.random.choice(len(X_train), labelled_instances, replace=False)
            y_train[random_unlabeled_points] = -1
            X = X_train
            y = y_train
            comite.add_model(flexCon.fit(X, y, option))
        with open('Comite_Heterogeneo.txt', 'a') as f:
            f.write('\n\nExecutando com KNN...\n\n')
        for i in list_knn_het:
            flexCon = SelfFlexCon(i)
            random_unlabeled_points = np.random.choice(len(X_train), labelled_instances, replace=False)
            y_train[random_unlabeled_points] = -1
            X = X_train
            y = y_train
            comite.add_model(flexCon.fit(X, y, option))

    else:
        print('Classificador não disponível! Insira outro...\n')
        print("Classificadores disponíveis:\n\n\t(1) Naive\n\t(2) Tree\n\t(3) KNN\n\t(4) Comite Heterogeneo\n\t(0) Sair\n\n")
        option = input('Informe o classificador: ')
    
    y_pred = comite.predict(digits_instances[random_unlabeled_points, :])
    y_true = digits_target_unlabelled_copy[random_unlabeled_points]

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

    elif option == 'Comite Heterogeneo' or option == '4':
        print('Salvando os resultados em um arquivo Comite_Heterogeneo.txt\n\n')
        print('Finalizando...')
        with open('Comite_Heterogeneo.txt', 'a') as f:
            f.write(
                f"\n\nACC: {round(accuracy_score(y_true, y_pred), 4) * 100}%\n"
                f'F1-Score: {round(f1_score(y_true, y_pred, average="macro"), 4) * 100}%\n'
                f"Motivo da finalização: {comite.ensemble[0].termination_condition_}\n"
                # f"Valor do teste estatístico é de {alpha}, significante? {alpha <= 0.05}\n"
            )
