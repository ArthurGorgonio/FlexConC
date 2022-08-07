from statistics import mode
from typing import List, NoReturn

import numpy as np
from sklearn.metrics import accuracy_score


class Ensemble:
    """
    Classe responsável por criar um cômite de classificadores e implementar
    seus métodos
    """

    def __init__(self, ssl_algorithm: callable, ssl_params):
        self.ensemble = []
        self.ssl_algorithm = ssl_algorithm
        self.ssl_params = ssl_params

    def add_classifier(self, classifier, need_train: bool = True):
        """
        Adiciona um classificador ao comitê, podendo já estar treinado
        ou não.

        Parameters
        ----------
        classifier : Classifer
            Classificador a ser adicionado no comitê.
        need_train : bool, optional
            Flag para indicar se é necessário treinar o classificador
            com o algoritmo semissupervisionado, por default True.
        """
        if need_train:
            self.ssl_params['base_estimator'] = classifier
            flexconc = self.ssl_algorithm(self.ssl_params)
        else:
            flexconc = classifier
        self.ensemble.append(flexconc)

    def add_fit_classifier(
        self,
        classifier,
        instances: np.ndarray,
        classes: np.ndarray
    ):
        """
        Treina um classificador e o adiciona no comitê já existente,
        evitando ter que treinar todo o comitê novamente.

        Parameters
        ----------
        classifier : Classifier
            Classificador a ser treinado e adicionado.
        instances : ndarray
            Instâncias para serem utilizadas no treinamento.
        classes : ndarray
            Rótulos dos instâncias.
        """
        classifier.fit(instances, classes)
        self.ensemble.append(classifier)

    def remover_classifier(self, classifier):
        """
        Remove um classificador específico do comitê atual.

        Parameters
        ----------
        classifier : Classificador
            Obejto classificador a ser removido do comitê.
        """
        self.ensemble.remove(classifier)

    def measure_classifier(self, instances: np.ndarray, classes: np.ndarray) -> List[float]:
        """
        Calcula a acurácia dos classificadores base do comitê.

        Parameters
        ----------
        instances : ndarray
            Instâncias para serem utilizadas no treinamento.
        classes : ndarray
            Rótulos dos instâncias.

        Returns
        -------
        List[float]
            Acurácia de cada classificador base do comitê.
        """
        measure_ensemble = []

        for classifier in self.ensemble:
            y_pred = self.predict_one_classifier(classifier, instances)
            measure_ensemble.append(accuracy_score(classes, y_pred))

        return measure_ensemble

    def drop_ensemble(self):
        """Esvazia o cômite de classificadores"""
        self.ensemble = []

    def fit_ensemble(self, instances: np.ndarray, classes: np.ndarray):
        """
        Treina todos os classificadores de uma vez, com um único
        conjunto de instâncias e classes. Essa função chama n vezes a
        função `fit_single_classifier`.

        Parameters
        ----------
        instances : ndarray
            Instâncias para serem utilizadas no treinamento.
        classes : ndarray
            Rótulos dos instâncias.
        """
        for classifier in self.ensemble:
            self.fit_single_classifier(classifier, instances, classes)

    def fit_single_classifier(
        self,
        classifier,
        instances: np.ndarray,
        classes: np.ndarray
    ):
        """
        Treina um classificador a partir de um conjunto de instâncias e
        seus respectivos rótulos

        Parameters
        ----------
        classifier : Classifier
            Classificador a ser treinado.
        instances : ndarray
            Instâncias para serem utilizadas no treinamento.
        classes : ndarray
            Rótulos dos instâncias.

        Returns
        -------
            Classificador treinado e pronto para ser utilizado para
            predizer novas instâncias da base de dados.
        """
        return classifier.fit(instances, classes)

    def predict_one_classifier(
        self,
        classifier,
        instances: np.ndarray
        ) -> List[int]:
        """
        Realiza a predição de um único classificador a partir de um
        grupo de instâncias.=

        Parameters
        ----------
        classifier : Classifier
            Classificador a ser utilizado para predizer as instâncias.
        instances : ndarray
            Instâncias para serem preditas.

        Returns
        -------
        List[int]
            Lista com os rótulos das instâncias que foram preditas pelo
            classificador
        """
        y_pred = classifier.predict(instances)

        return y_pred

    def predict(self, instances: np.ndarray) -> List[int]:
        """
        Realiza a predição de instâncias do comitê. A estratégia de
        agragação é a votação simples.

        Parameters
        ----------
        instances : np.ndarray
            Instâncias a serem avaliadas.

        Returns
        -------
        List[int]
            Predição do comitê de classificadores, a partir de uma
            votação simples para decidir os rótulos das instâncias.
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

    def swap(self, classifier: List, pos: List[int]) -> NoReturn:
        """
        Função para trocar um classificador do comitê por outro já
        treinado.

        Parameters
        ----------
        classifier : List
            Lista de classificadores, ou classificador, para serem
            incluídos no comitê.
        pos : List[int]
            Lista indicando quais classificadores do comitê devem ser
            substituídos pelos novos.

        Returns
        -------
        NoReturn
            _description_
        """
        if len(pos) == len(classifier):
            for i, j in zip(pos, classifier):
                self.ensemble[i] = classifier[j]
        else:
            for i in pos:
                self.ensemble[i] = classifier

    def partial_fit(
        self,
        instances: np.ndarray,
        classes: np.ndarray,
        labels=None,
        sample_weight=None
    ):
        """
        Função que realiza um treinamento parcial dos classificadores
        base do comitê.

        Parameters
        ----------
        instances : np.ndarray
            Instâncias a serem utilizadas no treinamento.
        classes : np.ndarray
            Rótulos das instâncias.
        """
        for model in self.ensemble:
            model.partial_fit(instances, classes, labels, sample_weight)

        return self

    def fit(
        self,
        instances: np.ndarray,
        classes: np.ndarray,
        labels=None,
        sample_weight=None
    ):
        """
        Função para fazer o treinamento dos classificadores do comitê.

        Parameters
        ----------
        instances : np.ndarray
            Instâncias a serem utilizadas no treinamento.
        classes : np.ndarray
            Rótulos das instâncias.
        """
        for model in self.ensemble:
            model.fit(instances, classes, labels, sample_weight)

        return self
