import argparse
import os
import warnings
from os import listdir
from os.path import isfile, join
from random import seed

import pandas as pd
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB as Naive

import src.utils as ut
from src.ssl.ensemble import Ensemble
from src.ssl.self_flexcon import SelfFlexCon

# from src.ssl.flexcon import BaseFlexConC
# from sklearn.semi_supervised import SelfTrainingClassifier
# from sklearn.base import BaseEstimator, MetaEstimatorMixin

cont = 0
quant = 10

# clas = SelfTrainingClassifier(MetaEstimatorMixin, BaseEstimator)
# base = clas.base_estimator
# cr = BaseFlexConC(base_estimator=base, cr=0.1, threshold=0.9, verbose=True)
# valor_cr, valor_threshold = cr.get_atributs
valor_cr = 0.05
valor_threshold = 0.95

for i in range(quant):
    
    cont += 1

    warnings.simplefilter("ignore")

    parser = argparse.ArgumentParser(description="Escolha um classificador para criar um cômite")
    parser.add_argument('classifier', metavar='c', type=int, help='Escolha um classificador para criar um cômite. Opções: 1 - Naive Bayes, 2 - Tree Decision, 3 - Knn, 4 - Heterogeneous')

    args = parser.parse_args()
    comite = Ensemble(SelfFlexCon)
    parent_dir = "path_for_results"

    # datasets = [f for f in listdir('datasets/') if isfile(join('datasets/', f))]
    # init_labelled = [0.05, 0.10, 0.15, 0.20, 0.25]
    datasets = ['Car.csv']
    init_labelled = [0.1]

    for dataset in datasets:
        path = os.path.join(parent_dir, dataset)
        os.makedirs(path, exist_ok=True)
        for labelled_level in init_labelled:
            fold_result = []
            df = pd.read_csv('datasets/'+dataset, header=0)
            seed(214)
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
                        print(f"\n\nO sistema irá selecionar instâncias da base {dataset}. Para o treinamento, será usado {round(labelled_level, 4) * 100}% das instâncias rotuladas de um total de {len(_instances)}. {cont}ª rodada de {quant}.\n\n")
                    instanciasRot = labelled_instances
                    instanciasRotPCento = (round(labelled_level, 4) * 100)
                    tInstanciasRot = "Instâncias rotuladas: " + str(labelled_instances)
                    tInstanciasRotPCento = " Usando: " + str(instanciasRotPCento) + "% das instâncias rotuladas "
                    if args.classifier == 1:
                        if(fold == 1):
                            fold += 1
                            with open(f'{path}/Comite_Naive_{round(labelled_level, 4) * 100} ({dataset}).txt', 'a') as f:
                                f.write(
                                    "-----------------------------------------------------------------------------------"
                                    f"\n|{tInstanciasRot.center(40)}"
                                    f"|{tInstanciasRotPCento.center(40)}|\n"
                                    f"|- - - - - - - - - ACC  - - - - - - - - -|"
                                    f"- - - - - - - - F1-Score - - - - - - - -|"
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
                                    "-----------------------------------------------------------------------------------"
                                    f"\n|{tInstanciasRot.center(40)}"
                                    f"|{tInstanciasRotPCento.center(40)}|\n"
                                    f"|- - - - - - - - - ACC  - - - - - - - - -|"
                                    f"- - - - - - - - F1-Score - - - - - - - -|"
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
                                    "-----------------------------------------------------------------------------------"
                                    f"\n|{tInstanciasRot.center(40)}"
                                    f"|{tInstanciasRotPCento.center(40)}|\n"
                                    f"|- - - - - - - - - ACC  - - - - - - - - -|"
                                    f"- - - - - - - - F1-Score - - - - - - - -|"
                                )
                        y = ut.select_labels(y_train, X_train, labelled_instances)
                        for i in ut.list_knn:
                            comite.add_classifier(i)
                        comite.fit_ensemble(X_train, y)

                    elif args.classifier == 4:
                        if(fold == 1):
                            fold += 1
                            with open(f'{path}/Comite_Heterogeneo_{round(labelled_level, 4) * 100} ({dataset}).txt', 'a') as f:
                                f.write(
                                    "-----------------------------------------------------------------------------------"
                                    f"\n|{tInstanciasRot.center(40)}"
                                    f"|{tInstanciasRotPCento.center(40)}|\n"
                                    f"|- - - - - - - - - ACC  - - - - - - - - -|"
                                    f"- - - - - - - - F1-Score - - - - - - - -|"
                                )
                        y = ut.select_labels(y_train, X_train, labelled_instances)
                        for i in range(9):
                            comite.add_classifier(Naive(var_smoothing=float(f'1e-{i}')))
                        for i in ut.list_tree:
                            comite.add_classifier(i)
                        for i in ut.list_knn:
                            comite.add_classifier(i)
                        comite.fit_ensemble(X_train, y)

                    y_pred = comite.predict(X_test)

                    fold_result.append(ut.result(args.classifier, dataset, y_test, y_pred, path, labelled_level))
            ut.calculateMeanStdev(fold_result, args.classifier, labelled_level, path, dataset, cont, quant, valor_cr, valor_threshold)