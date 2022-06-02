from statistics import mode

import numpy as np


class Ensemble:
    """
    Classe responsável por criar um cômite de classificadores e implementar
    seus métodos
    """

    def __init__(self, ssl_algorithm: callable):
        self.ensemble = []
        self.ssl_algorithm = ssl_algorithm

    def add_classifier(self, classifier):
        """
        Adiciona um novo classificador no cômite

        Args:
            classifier: Classificador
        """
        flexconc = self.ssl_algorithm(classifier)
        self.ensemble.append(flexconc)

    def remover_classifier(self, classifier):
        """
        Remove um classificador do cômite

        Args:
            classifier: Classificador
        """
        self.ensemble.remove(classifier)

    def measure_classifier(self, instances) -> list:
        """
        Calcula métrica de classificação

        Args:
            instances: instâncias da base de dados

        Returns:
            métrica de classificação do cômite
        """
        measure_ensemble = []

        for classifier in self.ensemble:
            measure_ensemble.append(
                self.predict_one_classifier(classifier, instances)
            )

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
