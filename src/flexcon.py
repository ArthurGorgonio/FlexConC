import warnings
from typing import List, Optional

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.utils import safe_mask

from src.utils import validate_estimator


class FlexConC(SelfTrainingClassifier):
    """
    Funcão do FleconC, responsável por classificar instâncias com base em
        modelos de aprendizado semisupervisionado

    Args:
        SelfTrainingClassifier
    """
    def __init__(
        self,
        base_estimator,
        cr=0.05,
        threshold=0.95,
        max_iter=100,
        verbose=False
    ):
        super().__init__(base_estimator=base_estimator,
                        threshold=threshold,
                        max_iter=max_iter)
        self._cr = cr
        self._init_acc = 0
        self.verbose = verbose
        self.old_selected = []
        self.selected = []
        self.dict_first = {}
        self.init_labeled_ = None
        self.labeled_iter_ = None
        self.transduction_ = None
        self.classes_ = None
        self.termination_condition_ = None
        self.pred_x_it = None
        self.cl_memory = None
        self.base_estimator_select_ = None
        self.base_estimator_ = clone(self.base_estimator)
        self.base_estimator_select_ = clone(self.base_estimator)
        self.n_iter_ = None


    def __str__(self):
        return (f'Classificador {self.base_estimator}\n'
                f'Outros Parâmetro:'
                f' CR: {self._cr}\t Threshold: {self.threshold}'
                f' Máximo IT: {self.max_iter}')


    def fit(self, X, y):
        """
        Fit self-training classifier using `X`, `y` as training data.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Array representing the data.
        y : {array-like, sparse matrix} of shape (n_samples,)
            Array representing the labels. Unlabeled samples should have the
            label -1.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = self._validate_data(
            X, y, accept_sparse=["csr", "csc", "lil", "dok"], force_all_finite=False
        )
        if self.base_estimator is None:
            raise ValueError("base_estimator cannot be None!")


        if not self.max_iter and self.max_iter < 0:
            raise ValueError(f"max_iter must be >= 0 or None, got {self.max_iter}")

        if not 0 <= self.threshold < 1:
            raise ValueError(f"threshold must be in [0,1), got {self.threshold}")

        if self.criterion not in ["threshold"]:
            raise ValueError(
                "criterion must be either 'threshold', "
                f"got {self.criterion}."
            )

        if y.dtype.kind in ["U", "S"]:
            raise ValueError(
                "y has dtype string. If you wish to predict on "
                "string targets, use dtype object, and use -1"
                " as the label for unlabeled samples."
            )

        has_label = y != -1

        self.cl_memory = [[0] * np.unique(y[has_label]) for _ in range(len(X))]

        if np.all(has_label):
            warnings.warn("y contains no unlabeled samples", UserWarning)

        init_acc = self.train_new_classifier(has_label, X, y)

        while not np.all(has_label) and (
            self.max_iter is None or self.n_iter_ < self.max_iter
        ):
            self.n_iter_ += 1
            self.base_estimator_.fit(
                X[safe_mask(X, has_label)], self.transduction_[has_label]
            )

            # Validate the fitted estimator since `predict_proba` can be
            # delegated to an underlying "final" fitted estimator as
            # generally done in meta-estimator or pipeline.
            validate_estimator(self.base_estimator_)

            # Predict on the unlabeled samples
            prob = self.base_estimator_.predict_proba(X[safe_mask(X, ~has_label)])
            pred = self.base_estimator_.classes_[np.argmax(prob, axis=1)]
            max_proba = np.max(prob, axis=1)

            if self.n_iter_ == 1:
                self.dict_first = self.storage_predict(
                    ids=np.nonzero(~has_label)[0],
                    confidence=max_proba,
                    classes=pred
                )
                # Select new labeled samples
                selected = max_proba >= self.threshold
            else:
                self.pred_x_it = self.storage_predict(
                    ids=np.nonzero(~has_label)[0],
                    confidence=max_proba,
                    classes=pred
                )
                selected, pred = self.rule_1()
                if not selected:
                    selected, pred  = self.rule_2()
                    if not selected:
                        selected, pred = self.rule_3()
                        if not selected:
                            selected, pred = self.rule_4()
                            if not selected:
                                self.threshold = np.max(max_proba)

            # Map selected indices into original array
            selected_full = np.nonzero(~has_label)[0][selected]
            # import ipdb; ipdb.sset_trace()
            self.update_memory(np.nonzero(~has_label)[0], pred)
            # Predict on the labeled samples
            try:
                # Traning model to classify the labeled samples
                self.base_estimator_select_.fit(
                    X[safe_mask(X, selected_full)], pred[selected]
                )

                local_acc = self.calc_local_measure(
                    X[safe_mask(X, self.init_labeled_)],
                    y[self.init_labeled_],
                    self.base_estimator_select_
                )
                print(f'Acurácia do novo classificador: {local_acc}')
            except ValueError:
                pass

            # Add newly labeled confident predictions to the dataset
        self.add_new_labeled(selected_full, selected, local_acc, init_acc, max_proba, pred, has_label)

        if self.n_iter_ == self.max_iter:
            self.termination_condition_ = "max_iter"
        if np.all(has_label):
            self.termination_condition_ = "all_labeled"

        self.base_estimator_.fit(
            X[safe_mask(X, has_label)], self.transduction_[has_label]
        )
        self.classes_ = self.base_estimator_.classes_
        return self

    @classmethod
    def calc_local_measure(cls, X, y_true, classifier):
        """
        Calcula o valor da acurácia do modelo

        Args:
            X: instâncias
            y_true: classes
            classifier: modelo

        Returns:
            Retorna a acurácia do modelo
        """
        y_pred = classifier.predict(X)
        return accuracy_score(y_true, y_pred)

    def new_threshold(self, local_measure, init_acc):
        """
        Responsável por calcular o novo limiar

        Args:
            local_measure: valor da acurácia do modelo treinado
            init_acc: valor da acurácia inicial
        """
        if local_measure > (init_acc + 0.01) and ((self.threshold - self._cr) > 0.0):
            self.threshold -= self._cr
        elif (local_measure < (init_acc - 0.01)) and ((self.threshold + self._cr) <= 1):
            self.threshold += self._cr
        else:
            pass

    def update_memory(self, instances: List, labels: List, weights: Optional[List] = None) -> None:
        """
        Atualiza a matriz de instâncias rotuladas

        Args:
            instances: instâncias
            labels: rotulos
            weights: Pesos de cada classe
        """
        if not weights:
            weights = [1 for _ in range(len(instances))]
            print(f'X = {instances}\n\n\n\nY = {labels}\n\n\n\n')
        for x, y, w in zip(instances, labels, weights):
            self.cl_memory[x][y] += w

    def remember(self, X: List) -> List:
        """
        Responsável por armazenar como está as instâncias dado um  momento no
        código

        Args:
            X: Lista com as instâncias

        Returns:
            A lista memorizada em um dado momento
        """
        y = [np.argmax(self.cl_memory[x]) for x in X]
        return y

    @classmethod
    def storage_predict(cls, ids, confidence, classes):
        """
        Responsável por armazenar o dicionário de dados da matriz

        Args:
            ids: indices de cada instância
            confidence: taxa de confiança para a classe destinada
            classes: indices das classes

        Returns:
            Retorna o dicionário com as classes das instâncias não rotuladas
        """
        dict = {}
        for i, conf, cl in zip(ids, confidence, classes):
            dict[i] = {}
            dict[i]['confidence'] = conf
            dict[i]['classes'] = cl
        return dict

    def rule_1(self):
        """
        regra responsável por verificar se as classes são iguais E uma das conf
            ianças preditas é maior que o limiar

        Returns:
        a lista correspondente pela condição
        """
        selected = []
        classes_selected = []
        for id in self.pred_x_it:
            if (self.dict_first[id]['confidence'] >= self.threshold
                and self.pred_x_it[id]['confidence'] >= self.threshold
                and self.dict_first[id]['classes'] == self.pred_x_it[id]['classes']):
                selected.append(id)
                classes_selected.append(self.dict_first[id]['classes'])
        return selected, classes_selected

    def rule_2(self):
        """
        regra responsável por verificar se as classes são iguais E uma das
        confianças preditas é maior que o limiar

        Returns:
        a lista correspondente pela condição
        """

        selected = []
        classes_selected = []
        for id in self.pred_x_it:
            if ((self.dict_first[id]['confidence'] >= self.threshold
                or self.pred_x_it[id]['confidence'] >= self.threshold)
                and self.dict_first[id]['classes'] == self.pred_x_it[id]['classes']):
                selected.append(id)
                classes_selected.append(self.dict_first[id]['classes'])
        return selected, classes_selected

    def rule_3(self):
        """
        regra responsável por verificar se as classes são diferentes E  as
        confianças preditas são maiores que o limiar

        Returns:
        a lista correspondente pela condição
        """
        selected = []
        for id in self.pred_x_it:
            if (self.dict_first[id]['classes'] != self.pred_x_it[id]['classes']
                and self.dict_first[id]['confidence'] >= self.threshold
                    and self.pred_x_it[id]['confidence'] >= self.threshold ):
                selected.append(id)
        return selected, self.remember(selected)

    def rule_4(self):
        """
        regra responsável por verificar se as classes são diferentes E uma das
        confianças preditas é maior que o limiar

        Returns:
        a lista correspondente pela condição
        """
        selected = []
        for id in self.pred_x_it:
            if (self.dict_first[id]['classes'] != self.pred_x_it[id]['classes']
                and (self.dict_first[id]['confidence'] >= self.threshold
                or self.pred_x_it[id]['confidence'] >= self.threshold)):
                selected.append(id)
        return selected, self.remember(selected)


    def train_new_classifier(self, has_label, X, y):
        """
        Responsável por treinar um classificador e mensurar
            a sua acertividade

        Args:
            has_label: lista com as instâncias rotuladas
            X: instâncias
            y: rótulos

        Returns:
            Acurácia do modelo
        """

        self.transduction_ = np.copy(y)
        self.labeled_iter_ = np.full_like(y, -1)
        self.labeled_iter_[has_label] = 0
        self.init_labeled_ = has_label.copy()

        self.n_iter_ = 0

        base_estimator_init = clone(self.base_estimator)
        # L0 - MODELO TREINADO E CLASSIFICADO COM L0
        base_estimator_init.fit(
            X[safe_mask(X, has_label)], self.transduction_[has_label]
        )
        # ACC EM L0 - RETORNA A EFICACIA DO MODELO
        init_acc = self.calc_local_measure(
                    X[safe_mask(X, self.init_labeled_)],
                    y[self.init_labeled_],
                    base_estimator_init
                    )
        print(f'Acurácia do novo classificador: {init_acc}')

        return init_acc


    def add_new_labeled(self, selected_full, selected, local_acc, init_acc, max_proba, pred, has_label):
        """
        Função que retorna as intâncias rotuladas

        Args:
            selected_full: lista com os indices das instâncias originais
            selected: lista das intâncias com acc acima do limiar
            local_acc: acurácia do modelo treinado com base na lista selected
            init_acc: acurácia do modelo treinado com base na lista selected_full
            max_proba: valores de probabilidade de predição das intâncias não rotuladas
            pred: predição das instâncias não rotuladas
            has_label: lista de instâncias rotuladas
        """
        self.transduction_[selected_full] = pred[selected]
        has_label[selected_full] = True
        self.labeled_iter_[selected_full] = self.n_iter_

        if selected_full.shape[0] > 0:
            # no changed labels
            self.new_threshold(local_acc, init_acc)
            self.termination_condition_ = "threshold_change"
        else:
            self.threshold = np.max(max_proba)
        if self.verbose:
            print(
                f"End of iteration {self.n_iter_},"
                f" added {selected} new labels."
            )
        else:
            print(
                f"End of iteration {self.n_iter_},"
                f" 0 new instaces are labelled."
            )
