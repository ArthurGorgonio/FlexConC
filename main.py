import warnings

import numpy as np
import pandas as pd

import src.utils as ut

from os import listdir
from os.path import isfile, join

from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB as Naive

from src.ssl.ensemble import Ensemble
from src.ssl.self_flexcon import SelfFlexCon

warnings.simplefilter("ignore")

comite = Ensemble(SelfFlexCon)

# datasets = path.list('datasets/')
# datasets = [f for f in listdir('datasets_bck/') if isfile(join('datasets_bck/', f))]
datasets = [f for f in listdir('datasets/') if isfile(join('datasets/', f))]
init_labelled = [0.05, 0.1, 0.15, 0.2]

for dataset in datasets:
    for labelled_level in init_labelled:
        # df = pd.read_csv('datasets_bck/'+dataset, header=0)
        df = pd.read_csv('datasets/'+dataset, header=0)
        kfold = StratifiedKFold(n_splits=10)
        fold = 1
        _instances = df.iloc[:,:-1].values #X
        _target_unlabelled = df.iloc[:,-1].values #Y
        # _target_unlabelled_copy = _target_unlabelled.copy()

        # DISPLAY QUE INFORMA PARA O USUÁRIO COMO PROCEDER
        print(f"\n\nO sistema irá selecionar instâncias da base {dataset}. Para o treinamento, será usado {round(labelled_level, 4) * 100}% das instâncias rotuladas de um total de {len(_instances)}.\n\n")
        print("Logo abaixo selecione qual classificador irá ser utilizado para criação do modelo preditivo.\n\n")
        print("Classificadores disponíveis:\n\n\t(1) Naive\n\t(2) Tree\n\t(3) KNN\n\t(4) Comite Heterogeneo\n\t(0) Sair\n\n")
        option = input('Informe o classificador: ')
        print('\nExecutando o treinamento...\n\n')
        for train, test in kfold.split(_instances, _target_unlabelled):
            X_train, X_test = _instances[train], _instances[test]
            y_train, y_test = _target_unlabelled[train], _target_unlabelled[test]
            labelled_instances = round(len(X_train)*labelled_level)

            if option == 'Sair' or option == '0':
                print('\nEncerrando...\n')
                exit()
            
            elif option == 'Naive' or option == '1':
                if(fold == 1):
                    fold += 1
                    with open(f'Comite_Naive_{round(labelled_level, 4) * 100}.txt', 'a') as f:
                        f.write(
                            f"Instâncias rotuladas: {labelled_instances}\n" 
                            f"Usando: {round(labelled_level, 4) * 100}% das instâncias rotuladas\n"
                        )
                y = ut.select_labels(y_train, X_train, labelled_instances)
                for i in range(9):
                    comite.add_classifier(Naive(var_smoothing=float(f'1e-{i}')))
                comite.fit_ensemble(X_train, y)

            elif option == 'Tree' or option == '2':
                if(fold == 1):
                    fold += 1
                    with open(f'Comite_Tree_{round(labelled_level, 4) * 100}.txt', 'a') as f:
                        f.write(
                            f"Instâncias rotuladas: {labelled_instances}\n" 
                            f"Usando: {round(labelled_level, 4) * 100}% das instâncias rotuladas\n"
                        )
                y = ut.select_labels(y_train, X_train, labelled_instances)
                for i in ut.list_tree:
                    comite.add_classifier(i)
                comite.fit_ensemble(X_train, y)

            elif option == 'KNN' or option == '3':
                if(fold == 1):
                    fold += 1
                    with open(f'Comite_KNN_{round(labelled_level, 4) * 100}.txt', 'a') as f:
                        f.write(
                            f"Instâncias rotuladas: {labelled_instances}\n" 
                            f"Usando: {round(labelled_level, 4) * 100}% das instâncias rotuladas\n"
                        )
                y = ut.select_labels(y_train, X_train, labelled_instances)
                for i in ut.list_knn:
                    comite.add_classifier(i)
                comite.fit_ensemble(X_train, y)

            elif option == 'Comite Heterogeneo' or option == '4':
                with open('Comite_Heterogeneo.txt', 'a') as f:
                    f.write(
                        f"Instâncias rotuladas: {labelled_instances}\n" 
                        f"Usando: {round(labelled_level, 4) * 100}% das instâncias rotuladas\n"
                    )
                    f.write('Comite heterogêneo selecionado...\n\n'
                            'Executando com Naive Bayes...\n\n')
                for i in range(5):
                    flexCon = SelfFlexCon(Naive(var_smoothing=float(f'1e-{i}')))
                    random_unlabeled_points = np.random.choice(len(X_train), labelled_instances, replace=False)
                    y_train[random_unlabeled_points] = -1
                    X = X_train 
                    y = y_train
                    comite.add_model(flexCon.fit(X, y))
                with open('Comite_Heterogeneo.txt', 'a') as f:
                    f.write('\n\nExecutando com Decision Tree...\n\n')
                for i in ut.list_tree_het:
                    flexCon = SelfFlexCon(i)
                    random_unlabeled_points = np.random.choice(len(X_train), labelled_instances, replace=False)
                    y_train[random_unlabeled_points] = -1
                    X = X_train
                    y = y_train
                    comite.add_model(flexCon.fit(X, y))
                with open('Comite_Heterogeneo.txt', 'a') as f:
                    f.write('\n\nExecutando com KNN...\n\n')
                for i in ut.list_knn_het:
                    flexCon = SelfFlexCon(i)
                    random_unlabeled_points = np.random.choice(len(X_train), labelled_instances, replace=False)
                    y_train[random_unlabeled_points] = -1
                    X = X_train
                    y = y_train
                    comite.add_model(flexCon.fit(X, y))

            else:
                print('Opção inválida! Escolha corretamente...\n')
                exit()
            
            y_pred = comite.predict(X_test)

            ut.result(option, y_test, y_pred, comite, labelled_level)