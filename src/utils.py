from nis import match
from statistics import mean, stdev

import numpy as np
# from ipdb import set_trace
from sklearn.metrics import accuracy_score, f1_score

# from pushbullet.pushbullet import PushBullet
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB as Naive
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as Tree

# CONEXÃO COM PUSHBULLET
# apiKey = "o.uV6lySsrOlDbTyo0OJg9Jzh6SOZcWCTz"
# p = PushBullet(apiKey)
# devices = p.getDevices()

# LISTA DE CLASSIFICADORES
list_tree = [
    Tree(),
    Tree(splitter="random"),
    Tree(max_features=None),
    Tree(criterion="entropy"),
    Tree(criterion="entropy", splitter="random"),
    Tree(criterion="entropy", max_features=None),
    Tree(criterion="entropy", max_features=None, splitter="random"),
    Tree(criterion="entropy", max_features='sqrt', splitter="random"),
    Tree(max_features='sqrt', splitter="random"),
    Tree(max_features=None, splitter="random")]

list_knn= [
    KNN(n_neighbors=4, weights='distance'),KNN(n_neighbors=4),
    KNN(n_neighbors=5, weights='distance'),KNN(n_neighbors=5),
    KNN(n_neighbors=6, weights='distance'),KNN(n_neighbors=6),
    KNN(n_neighbors=7, weights='distance'),KNN(n_neighbors=7),
    KNN(n_neighbors=8, weights='distance'),KNN(n_neighbors=8)]

def validate_estimator(estimator):
    """Make sure that an estimator implements the necessary methods."""
    if not hasattr(estimator, "predict_proba"):
        msg = "base_estimator ({}) should implement predict_proba!"
        raise ValueError(msg.format(type(estimator).__name__))

def cross_validation(k):
    try:
        return StratifiedKFold(n_splits=k)
    except ValueError:
        print("Please put a number > 2")

def select_labels(y_train, X_train, labelled_instances):
    """
    Responsável por converter o array de rótulos das instâncias com base
    nas instâncias selecionadas randomicamente
    Args:
        y_train (Array): Classes usadas no treinamento
        X_train (Array): Instâncias
        labelled_instances (Array): Quantidade de instâncias rotuladas

    Returns:
        Retorna o array de classes com base nos rótulos das instâncias selecionadas
    """
    # set_trace()
    count = 0
    labels = np.unique(y_train)
    if -1 in labels:
        labels = list(filter(lambda result: result != -1, labels))
    while count != len(labels):
        count = 0
        instances = []
        random_unlabeled_points = np.random.choice(len(X_train), labelled_instances, replace=False)
        for instance in random_unlabeled_points:
            instances.append(y_train[instance])
        for label in labels:
            if label in instances: count += 1
    mask = np.ones(len(X_train), bool)
    mask[random_unlabeled_points] = 0
    y_train[mask] = -1
    return y_train

def result(option, dataset, y_test, y_pred, path, labelled_level, cr, threshold, fold_result, rounds):
    """
    Responsável por salvar os outputs dos cômites em arquivos
    Args:
        option (int): Opção de escolha do usuário
        dataset (string): Base de dados nome
        y_test (Array): Rótulos usadas para testes
        y_pred (Array): Rótulos predizidos pelo modelo
        comite (): Cômite de classificadores
        labelled_level (float): % que foi selecionada na iteração
    """
    acc = round(accuracy_score(y_test, y_pred) * 100, 4)
    f1  = round(f1_score(y_test, y_pred, average="macro") * 100, 4)
    tAcc = str(acc) + "%"
    tF1Score = str(f1) + "%"
    cr = cr
    threshold = threshold
    rounds = rounds
    average = ''
    standard_deviation = ''
    if len(fold_result) == 10:
        average = round(mean(fold_result), 4)
        standard_deviation = round(stdev(fold_result), 4)
    if option == 1:
        print(f'Salvando os resultados em arquivos Comite_Naive_{round(labelled_level, 4) * 100} ({dataset}).txt e Comite_Naive_{round(labelled_level, 4) * 100} ({dataset}).csv\n\n')
        # print('Enviando notificação push...')
        # p.pushNote(devices[1]["iden"], f"ACC: {round(accuracy_score(y_test, y_pred), 4) * 100}%", f'Comite_Naive_{round(labelled_level, 4) * 100} ({dataset}).txt')
        with open(f'{path}/Comite_Naive_{round(labelled_level, 4) * 100} ({dataset}).txt', 'a') as f:
            f.write(
                #ACC
                f"\n|{tAcc.center(28)}|"
                #F1-Score
                f'{tF1Score.center(28)}|'
                # f"Motivo da finalização: {comite.ensemble[0].termination_condition_}\n"
                # f"Valor do teste estatístico é de {alpha}, significante? {alpha <= 0.05}\n"
            )
        with open(f'{path}/Comite_Naive_{round(labelled_level, 4) * 100} ({dataset}).csv', 'a') as f:
            f.write(
                #ROUNDS
                f'\n{rounds},'
                #DATA-SET
                f'"{dataset}",'
                #LABELLED_LEVEL
                f'{labelled_level},'
                #CR
                f'{cr},'
                #THRESHOLD
                f'{threshold},'
                #ACC
                f'{acc},'
                #F1-Score
                f'{f1},'
                # AVERAGE
                f'{average},'
                #STANDARD_DEVIATION
                f'{standard_deviation}'
                # f"Motivo da finalização: {comite.ensemble[0].termination_condition_}\n"
                # f"Valor do teste estatístico é de {alpha}, significante? {alpha <= 0.05}\n"
            )
        return acc

    elif option == 2:
        print(f'Salvando os resultados em arquivos Comite_Tree_{round(labelled_level, 4) * 100} ({dataset}).txt e Comite_Tree_{round(labelled_level, 4) * 100} ({dataset}).csv\n\n')
        # print('Enviando notificação push...')
        # p.pushNote(devices[1]["iden"], f"ACC: {round(accuracy_score(y_test, y_pred), 4) * 100}%", f'Comite_Tree_{round(labelled_level, 4) * 100} ({dataset}).txt')
        with open(f'{path}/Comite_Tree_{round(labelled_level, 4) * 100} ({dataset}).txt', 'a') as f:
            f.write(
                #ACC
                f"\n|{tAcc.center(28)}|"
                #F1-Score
                f'{tF1Score.center(28)}|'
                # f"Motivo da finalização: {comite.ensemble[0].termination_condition_}\n"
                # f"Valor do teste estatístico é de {alpha}, significante? {alpha <= 0.05}\n"
            )
        with open(f'{path}/Comite_Tree_{round(labelled_level, 4) * 100} ({dataset}).csv', 'a') as f:
            f.write(
                #ROUNDS
                f'\n{rounds},'
                #DATA-SET
                f'"{dataset}",'
                #LABELLED_LEVEL
                f'{labelled_level},'
                #CR
                f'{cr},'
                #THRESHOLD
                f'{threshold},'
                #ACC
                f'{acc},'
                #F1-Score
                f'{f1},'
                # AVERAGE
                f'{average},'
                #STANDARD_DEVIATION
                f'{standard_deviation}'
                # f"Motivo da finalização: {comite.ensemble[0].termination_condition_}\n"
                # f"Valor do teste estatístico é de {alpha}, significante? {alpha <= 0.05}\n"
            )
        return acc

    elif option == 3:
        print(f'Salvando os resultados em arquivos Comite_KNN_{round(labelled_level, 4) * 100} ({dataset}).txt e Comite_KNN_{round(labelled_level, 4) * 100} ({dataset}).csv\n\n')
        # print('Enviando notificação push...')
        # TODO: FALTA PEGAR CADA RESULTADO PREENCHER UM ARRAY REALIZAR A MÉDIA E AI SIM ENVIAR A MSG
        # p.pushNote(devices[1]["iden"], f"ACC: {round(accuracy_score(y_test, y_pred), 4) * 100}%", f'Comite_KNN_{round(labelled_level, 4) * 100} ({dataset}).txt')
        with open(f'{path}/Comite_KNN_{round(labelled_level, 4) * 100} ({dataset}).txt', 'a') as f:
            f.write(
                #ACC
                f"\n|{tAcc.center(28)}|"
                #F1-Score
                f'{tF1Score.center(28)}|'
                # f"Motivo da finalização: {comite.ensemble[0].termination_condition_}\n"
                # f"Valor do teste estatístico é de {alpha}, significante? {alpha <= 0.05}\n"
            )
        with open(f'{path}/Comite_KNN_{round(labelled_level, 4) * 100} ({dataset}).csv', 'a') as f:
            f.write(
                #ROUNDS
                f'\n{rounds},'
                #DATA-SET
                f'"{dataset}",'
                #LABELLED_LEVEL
                f'{labelled_level},'
                #CR
                f'{cr},'
                #THRESHOLD
                f'{threshold},'
                #ACC
                f'{acc},'
                #F1-Score
                f'{f1},'
                # AVERAGE
                f'{average},'
                #STANDARD_DEVIATION
                f'{standard_deviation}'
                # f"Motivo da finalização: {comite.ensemble[0].termination_condition_}\n"
                # f"Valor do teste estatístico é de {alpha}, significante? {alpha <= 0.05}\n"
            )
        return acc

    elif option == 4:
        print(f'Salvando os resultados em arquivos Comite_Heterogeneo_{round(labelled_level, 4) * 100} ({dataset}).txt e Comite_Heterogeneo_{round(labelled_level, 4) * 100} ({dataset}).csv\n\n')
        # print('Enviando notificação push...')
        # p.pushNote(devices[1]["iden"], f"ACC: {round(accuracy_score(y_test, y_pred), 4) * 100}%", f'Comite_KNN_{round(labelled_level, 4) * 100} ({dataset}).txt')
        with open(f'{path}/Comite_Heterogeneo_{round(labelled_level, 4) * 100} ({dataset}).txt', 'a') as f:
            f.write(
                #ACC
                f"\n|{tAcc.center(28)}|"
                #F1-Score
                f'{tF1Score.center(28)}|'
                # f"Motivo da finalização: {comite.ensemble[0].termination_condition_}\n"
                # f"Valor do teste estatístico é de {alpha}, significante? {alpha <= 0.05}\n"
            )
        with open(f'{path}/Comite_Heterogeneo_{round(labelled_level, 4) * 100} ({dataset}).csv', 'a') as f:
            f.write(
                #ROUNDS
                f'\n{rounds},'
                #DATA-SET
                f'"{dataset}",'
                #LABELLED_LEVEL
                f'{labelled_level},'
                #CR
                f'{cr},'
                #THRESHOLD
                f'{threshold},'
                #ACC
                f'{acc},'
                #F1-Score
                f'{f1},'
                # AVERAGE
                f'{average},'
                #STANDARD_DEVIATION
                f'{standard_deviation}'
                # f"Motivo da finalização: {comite.ensemble[0].termination_condition_}\n"
                # f"Valor do teste estatístico é de {alpha}, significante? {alpha <= 0.05}\n"
            )
        return acc

def calculateMeanStdev(fold_result, option, labelled_level, path, dataset, cr, threshold):
    media = round(mean(fold_result), 4)
    dPadrao = round(stdev(fold_result), 4)
    tMedia = "Média: " + str(media) + "%"
    tDesPadr = "Désvio padrão: " + str(dPadrao)
    tCR = "CR = " + str(cr)
    tTH = "THRESHOLD = " + str(threshold)
    if option == 1:
        with open(f'{path}/Comite_Naive_{round(labelled_level, 4) * 100} ({dataset}).txt', 'a') as f:
            f.write(
                f'\n|{tMedia.center(28)}|{tDesPadr.center(28)}|\n'
                f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n"
                f'|{tCR.center(28)}|{tTH.center(28)}|\n'
                f"-----------------------------------------------------------\n\n"
            )
    elif option == 2:
        with open(f'{path}/Comite_Tree_{round(labelled_level, 4) * 100} ({dataset}).txt', 'a') as f:
            f.write(
                f'\n|{tMedia.center(28)}|{tDesPadr.center(28)}|\n'
                f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n"
                f'|{tCR.center(28)}|{tTH.center(28)}|\n'
                f"-----------------------------------------------------------\n\n"
            )
    elif option == 3:
        with open(f'{path}/Comite_KNN_{round(labelled_level, 4) * 100} ({dataset}).txt', 'a') as f:
            f.write(
                f'\n|{tMedia.center(28)}|{tDesPadr.center(28)}|\n'
                f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n"
                f'|{tCR.center(28)}|{tTH.center(28)}|\n'
                f"-----------------------------------------------------------\n\n"
            )
    elif option == 4:
        with open(f'{path}/Comite_Heterogeneo_{round(labelled_level, 4) * 100} ({dataset}).txt', 'a') as f:
            f.write(
                f'\n|{tMedia.center(28)}|{tDesPadr.center(28)}|\n'
                f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n"
                f'|{tCR.center(28)}|{tTH.center(28)}|\n'
                f"-----------------------------------------------------------\n\n"
            )