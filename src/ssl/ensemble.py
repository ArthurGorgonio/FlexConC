from statistics import mode

import numpy as np
from sklearn.metrics import accuracy_score


class Ensemble:
    """
    Classe responsável por criar um cômite de classificadores e implementar
    seus métodos
    """

    def __init__(self, ssl_algorithm: callable):
        self.ensemble = []
        self.ssl_algorithm = ssl_algorithm

    def add_classifier(self, classifier, need_train: bool = True):
        """
        Adiciona um novo classificador no cômite

        Args:
            classifier: Classificador
        """

        if need_train:
            flexconc = self.ssl_algorithm(classifier)
        else:
            flexconc = classifier
        self.ensemble.append(flexconc)

    def add_fit_classifier(self, classifier, instances, labels):
        """
        Adiciona um novo classificador durante em um comitê já existente.

        Args:
            classifier: Classificador a ser treinado e adicionado
            instances: Instâncias
            labels: Rótulos
        """
        classifier.fit(instances, labels)
        self.ensemble.append(classifier)

    def remover_classifier(self, classifier):
        """
        Remove um classificador do cômite

        Args:
            classifier: Classificador
        """
        self.ensemble.remove(classifier)

    def measure_classifier(self, instances, labels) -> list:
        """
        Calcula métrica de classificação

        Args:
            instances: instâncias da base de dados

        Returns:
            métrica de classificação do cômite
        """
        measure_ensemble = []

        for classifier in self.ensemble:
            y_pred = self.predict_one_classifier(classifier, instances)
            measure_ensemble.append(accuracy_score(labels, y_pred))

        return measure_ensemble

    def drop_ensemble(self):
        """
        Esvazia o cômite de classificadores
        """
        self.ensemble = []

    def fit_ensemble(self, instances, classes):
        """
        Treina os classificadores presentes no cômite
        """

        for classifier in self.ensemble:
            self.fit_single_classifier(classifier, instances, classes)

    def fit_single_classifier(self, classifier, instances, classes):
        """
        Treinar cada classificador iterativamente

        Args:
            classifier: classificador do cômite
            instances: instâncias da base de dados
            classes: classes da base de dados
        """

        return classifier.fit(instances, classes)

    def predict_one_classifier(self, classifier, instances):
        """
        Retorna a predição de um classificador
        Args:
            classifier: classificador
            instances: instâncias da base de dados
        """
        y_pred = classifier.predict(instances)

        return y_pred

    def predict(self, instances):
        """
        Retorna a predição mais comum entre as instâncias
        Args:
            instances: instâncias da base de dados
        """
        y_pred = np.array([], dtype="int64")

        for instance in instances:
            pred = []

            for classifier in self.ensemble:
                pred.append(
                    classifier.predict(instance.reshape(1, -1)).tolist()[0]
                )
            y_pred = np.append(y_pred, mode(pred))

        return y_pred

    def swap(self, classifier, pos: List) -> None:
        if len(pos) == len(classifier):
            for i, j in zip(pos, classifier):
                self.ensemble[i] = classifier[j]
        else:
            for i in pos:
                self.ensemble[i] = classifier
