from nis import match

import numpy as np
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
list_tree_het = [
    Tree(criterion="entropy"), Tree(), 
    Tree(criterion="entropy", max_features="log2"),
    Tree(criterion="entropy", max_features='auto'), Tree(max_features='auto')]

list_knn_het = [
    KNN(n_neighbors=1, weights='distance'),KNN(n_neighbors=1),
    KNN(n_neighbors=2, weights='distance'),KNN(n_neighbors=2),
    KNN(n_neighbors=3, weights='distance')]

list_tree = [
    Tree(criterion="entropy"), Tree(), 
    Tree(criterion="entropy", max_features=1), Tree(max_features=1),
    Tree(criterion="entropy", max_features="auto"), Tree(max_features='auto'),
    Tree(criterion="entropy", max_features='auto', splitter="random"), Tree(max_features='auto', splitter="random"),
    Tree(splitter="random"), Tree(criterion="entropy",splitter="random")]

# list_knn= [
#     KNN(n_neighbors=10, weights='distance'),KNN(n_neighbors=11, weights='distance'),
#     KNN(n_neighbors=12, weights='distance'),KNN(n_neighbors=13, weights='distance'),
#     KNN(n_neighbors=14, weights='distance'),KNN(n_neighbors=11),KNN(n_neighbors=12),
#     KNN(n_neighbors=13),KNN(n_neighbors=14)]

list_knn= [
    KNN(n_neighbors=1, weights='distance'),KNN(n_neighbors=1),
    KNN(n_neighbors=2, weights='distance'),KNN(n_neighbors=2),
    KNN(n_neighbors=3, weights='distance'),KNN(n_neighbors=3),
    KNN(n_neighbors=4, weights='distance'),KNN(n_neighbors=4),
    KNN(n_neighbors=5, weights='distance'),KNN(n_neighbors=5)]

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
    mask = np.ones(len(X_train), np.bool)
    mask[random_unlabeled_points] = 0
    y_train[mask] = -1
    return y_train

def result(option, dataset, y_test, y_pred, path, labelled_level):
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
    if option == 1:
        print(f'Salvando os resultados em um arquivo Comite_Naive_{round(labelled_level, 4) * 100} ({dataset}).txt\n\n')
        print('Finalizando...')
        # print('Enviando notificação push...')
        # p.pushNote(devices[1]["iden"], f"ACC: {round(accuracy_score(y_test, y_pred), 4) * 100}%", f'Comite_Naive_{round(labelled_level, 4) * 100} ({dataset}).txt')
        with open(f'{path}/Comite_Naive_{round(labelled_level, 4) * 100} ({dataset}).txt', 'a') as f:
            f.write(
                f"\n\nACC: {round(accuracy_score(y_test, y_pred), 4) * 100}%\n"
                f'F1-Score: {round(f1_score(y_test, y_pred, average="macro"), 4) * 100}%\n'
                # f"Motivo da finalização: {comite.ensemble[0].termination_condition_}\n"
                # f"Valor do teste estatístico é de {alpha}, significante? {alpha <= 0.05}\n"
            )

    elif option == 2:
        print(f'Salvando os resultados em um arquivo Comite_Tree_{round(labelled_level, 4) * 100} ({dataset}).txt\n\n')
        print('Finalizando...')
        # print('Enviando notificação push...')
        # p.pushNote(devices[1]["iden"], f"ACC: {round(accuracy_score(y_test, y_pred), 4) * 100}%", f'Comite_Tree_{round(labelled_level, 4) * 100} ({dataset}).txt')
        with open(f'{path}/Comite_Tree_{round(labelled_level, 4) * 100} ({dataset}).txt', 'a') as f:
            f.write(
                f"\n\nACC: {round(accuracy_score(y_test, y_pred), 4) * 100}%\n"
                f'F1-Score: {round(f1_score(y_test, y_pred, average="macro"), 4) * 100}%\n'
                # f"Motivo da finalização: {comite.ensemble[0].termination_condition_}\n"
                # f"Valor do teste estatístico é de {alpha}, significante? {alpha <= 0.05}\n"
            )

    elif option == 3:
        print(f'Salvando os resultados em um arquivo Comite_KNN_{round(labelled_level, 4) * 100} ({dataset}).txt\n\n')
        print('Finalizando...')
        # print('Enviando notificação push...')
        # TODO: FALTA PEGAR CADA RESULTADO PREENCHER UM ARRAY REALIZAR A MÉDIA E AI SIM ENVIAR A MSG
        # p.pushNote(devices[1]["iden"], f"ACC: {round(accuracy_score(y_test, y_pred), 4) * 100}%", f'Comite_KNN_{round(labelled_level, 4) * 100} ({dataset}).txt')
        with open(f'{path}/Comite_KNN_{round(labelled_level, 4) * 100} ({dataset}).txt', 'a') as f:
            f.write(
                f"\n\nACC: {round(accuracy_score(y_test, y_pred), 4) * 100}%\n"
                f'F1-Score: {round(f1_score(y_test, y_pred, average="macro"), 4) * 100}%\n'
                # f"Motivo da finalização: {comite.ensemble[0].termination_condition_}\n"
                # f"Valor do teste estatístico é de {alpha}, significante? {alpha <= 0.05}\n"
            )

    elif option == 4:
        print(f'Salvando os resultados em um arquivo Comite_Heterogeneo_{round(labelled_level, 4) * 100} ({dataset}).txt\n\n')
        print('Finalizando...')
        # print('Enviando notificação push...')
        # p.pushNote(devices[1]["iden"], f"ACC: {round(accuracy_score(y_test, y_pred), 4) * 100}%", f'Comite_KNN_{round(labelled_level, 4) * 100} ({dataset}).txt')
        with open(f'{path}/Comite_Heterogeneo_{round(labelled_level, 4) * 100} ({dataset}).txt', 'a') as f:
            f.write(
                f"\n\nACC: {round(accuracy_score(y_test, y_pred), 4) * 100}%\n"
                f'F1-Score: {round(f1_score(y_test, y_pred, average="macro"), 4) * 100}%\n'
                # f"Motivo da finalização: {comite.ensemble[0].termination_condition_}\n"
                # f"Valor do teste estatístico é de {alpha}, significante? {alpha <= 0.05}\n"
            )
