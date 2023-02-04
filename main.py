import argparse
import os
import warnings
from os import listdir
from os.path import isfile, join

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB as Naive

import src.utils as ut
from src.ssl.ensemble import Ensemble
from src.ssl.self_flexcon import SelfFlexCon

warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description="Escolha um classificador para criar um cômite")
parser.add_argument('classifier', metavar='c', type=int, help='Escolha um classificador para criar um cômite. Opções: 1 - Naive Bayes, 2 - Tree Decision, 3 - Knn, 4 - Heterogeneous')

args = parser.parse_args()
comite = Ensemble(SelfFlexCon)
parent_dir = "/home/gabriel/MyProjects/Python_projects/FlexConC/results"

datasets = [f for f in listdir('datasets/') if isfile(join('datasets/', f))]
init_labelled = [0.2]
# 0.05, 0.1, 0.15,
for dataset in datasets:
    path = os.path.join(parent_dir, dataset)
    os.mkdir(path)
    for labelled_level in init_labelled:
        df = pd.read_csv('datasets/'+dataset, header=0)
        kfold = StratifiedKFold(n_splits=10)
        fold = 1
        flag = 1
        _instances = df.iloc[:,:-1].values #X
        _target_unlabelled = df.iloc[:,-1].values #Y
        # _target_unlabelled_copy = _target_unlabelled.copy()

        for train, test in kfold.split(_instances, _target_unlabelled):
            X_train, X_test = _instances[train], _instances[test]
            y_train, y_test = _target_unlabelled[train], _target_unlabelled[test]
            labelled_instances = round(len(X_train)*labelled_level)

            if args.classifier != 1 and args.classifier != 2 and args.classifier != 3 and args.classifier != 4:
                print('\nOpção inválida! Escolha corretamente...\nOpções: 1 - Naive Bayes, 2 - Tree Decision, 3 - Knn, 4 - Heterogeneous\nEx: python main.py 1\n')
                exit()
            else:
                # DISPLAY QUE INFORMA PARA O USUÁRIO COMO PROCEDER
                if(flag == 1):
                    flag += 1
                    print(f"\n\nO sistema irá selecionar instâncias da base {dataset}. Para o treinamento, será usado {round(labelled_level, 4) * 100}% das instâncias rotuladas de um total de {len(_instances)}.\n\n")
                if args.classifier == 1:
                    if(fold == 1):
                        fold += 1
                        with open(f'{path}/Comite_Naive_{round(labelled_level, 4) * 100} ({dataset}).txt', 'a') as f:
                            f.write(
                                f"Instâncias rotuladas: {labelled_instances}\n" 
                                f"Usando: {round(labelled_level, 4) * 100}% das instâncias rotuladas\n"
                            )
                    y = ut.select_labels(y_train, X_train, labelled_instances)
                    for i in range(9):
                        comite.add_classifier(Naive(var_smoothing=float(f'1e-{i}')))
                    comite.fit_ensemble(X_train, y)
                elif args.classifier == 2:
                    if(fold == 1):
                        fold += 1
                        with open(f'{path}/Comite_Tree_{round(labelled_level, 4) * 100} ({dataset}).txt', 'a') as f:
                            f.write(
                                f"Instâncias rotuladas: {labelled_instances}\n" 
                                f"Usando: {round(labelled_level, 4) * 100}% das instâncias rotuladas\n"
                            )
                    y = ut.select_labels(y_train, X_train, labelled_instances)
                    for i in ut.list_tree:
                        comite.add_classifier(i)
                    comite.fit_ensemble(X_train, y)

                elif args.classifier == 3:
                    if(fold == 1):
                        fold += 1
                        with open(f'{path}/Comite_KNN_{round(labelled_level, 4) * 100} ({dataset}).txt', 'a') as f:
                            f.write(
                                f"Instâncias rotuladas: {labelled_instances}\n" 
                                f"Usando: {round(labelled_level, 4) * 100}% das instâncias rotuladas\n"
                            )
                    y = ut.select_labels(y_train, X_train, labelled_instances)
                    for i in ut.list_knn:
                        comite.add_classifier(i)
                    comite.fit_ensemble(X_train, y)

                elif args.classifier == 4:
                    if(fold == 1):
                        fold += 1
                        with open(f'Comite_Heterogeneo_{round(labelled_level, 4) * 100} ({dataset}).txt', 'a') as f:
                            f.write(
                                f"Instâncias rotuladas: {labelled_instances}\n" 
                                f"Usando: {round(labelled_level, 4) * 100}% das instâncias rotuladas\n"
                            )
                    y = ut.select_labels(y_train, X_train, labelled_instances)
                    for i in range(5):
                        comite.add_classifier(Naive(var_smoothing=float(f'1e-{i}')))
                    comite.fit_ensemble(X_train, y)
                    y = ut.select_labels(y_train, X_train, labelled_instances)
                    for i in ut.list_tree_het:
                        comite.add_classifier(i)
                    comite.fit_ensemble(X_train, y)
                    y = ut.select_labels(y_train, X_train, labelled_instances)
                    for i in ut.list_knn_het:
                        comite.add_classifier(i)
                    comite.fit_ensemble(X_train, y)
                
                y_pred = comite.predict(X_test)

                ut.result(args.classifier, dataset, y_test, y_pred, path, labelled_level)