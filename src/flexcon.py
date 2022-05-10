import warnings
from typing import Dict, List, Optional

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
    """

    def __init__(self, base_estimator, cr=0.05, threshold=0.95, verbose=False):
        super().__init__(
            base_estimator=base_estimator, threshold=threshold, max_iter=100
        )
        self.cr: float = cr
        self.verbose = verbose
        self.old_selected: List = []
        self.dict_first: Dict = {}
        self.init_labeled_: List = []
        self.labeled_iter_: List = []
        self.transduction_: List = []
        self.classes_: List = []
        self.termination_condition_ = ""
        self.pred_x_it: Dict = {}
        self.cl_memory: List = []
        self.base_estimator_ = clone(self.base_estimator)
        self.base_estimator_select_ = clone(self.base_estimator)

    def __str__(self):
        return (
            f"Classificador {self.base_estimator}\n"
            f"Outros Parâmetro:"
            f" CR: {self.cr}\t Threshold: {self.threshold}"
            f" Máximo IT: {self.max_iter}"
        )

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
            X,
            y,
            accept_sparse=["csr", "csc", "lil", "dok"],
            force_all_finite=False,
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
        old_selected = []
        old_pred = []
        self.n_iter_ = 0

        while not np.all(has_label) and (
            self.max_iter is None or self.n_iter_ <= self.max_iter
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
            prob = self.base_estimator_.predict_proba(
                X[safe_mask(X, ~has_label)]
            )
            pred = self.base_estimator_.classes_[np.argmax(prob, axis=1)]
            max_proba = np.max(prob, axis=1)

            if self.n_iter_ > 1:
                self.pred_x_it = self.storage_predict(
                    idx=np.nonzero(~has_label)[0],
                    confidence=max_proba,
                    classes=pred,
                )

                pseudo_ids = np.nonzero(~has_label)[0].tolist()
                selected_full, pred_full = self.select_instances_by_rules()

                if not selected_full.size:
                    self.threshold = np.max(max_proba)
                    selected_full, pred_full = self.select_instances_by_rules()
                selected = [pseudo_ids.index(inst) for inst in selected_full]
                pred[selected] = pred_full
            else:
                self.dict_first = self.storage_predict(
                    idx=np.nonzero(~has_label)[0],
                    confidence=max_proba,
                    classes=pred,
                )
                # Select new labeled samples
                selected = max_proba >= self.threshold
                # Map selected indices into original array
                selected_full = np.nonzero(~has_label)[0][selected]

            # Add newly labeled confident predictions to the dataset
            has_label[selected_full] = True

            self.update_memory(np.nonzero(~has_label)[0], pred)
            # Predict on the labeled samples
            try:
                if old_selected:
                    selected_full = np.array(old_selected + selected_full.tolist())
                    selected = old_selected + selected
                    print(f'Selected_FULL:\n{selected_full}\n{"="*30}\n\nSelected:\n{selected}\n{"-"*30}\n\n\n')
                    old_selected = []
                    old_pred = []
                    pass
                    # WIP - pred bugada!
                    self.base_estimator_select_.fit(
                        X[selected_full], pred[selected]
                    )
                else:
                    # Traning model to classify the labeled samples
                    self.base_estimator_select_.fit(
                        X[selected_full], pred[selected]
                    )
                #WIP

                local_acc = self.calc_local_measure(
                    X[safe_mask(X, self.init_labeled_)],
                    y[self.init_labeled_],
                    self.base_estimator_select_,
                )
                self.add_new_labeled(
                    selected_full,
                    selected,
                    local_acc,
                    init_acc,
                    max_proba,
                    pred,
                )
            except ValueError:
                old_selected = selected_full.tolist()
                old_pred = pred[selected].tolist()

        if self.n_iter_ == self.max_iter:
            self.termination_condition_ = "max_iter"

        if np.all(has_label):
            self.termination_condition_ = "all_labeled"

        self.base_estimator_.fit(
            X[safe_mask(X, has_label)], self.transduction_[has_label]
        )
        self.classes_ = self.base_estimator_.classes_

        return self

    def calc_local_measure(self, X, y_true, classifier):
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
        if local_measure > (init_acc + 0.01) and (
            (self.threshold - self.cr) > 0.0
        ):
            self.threshold -= self.cr
        elif (local_measure < (init_acc - 0.01)) and (
            (self.threshold + self.cr) <= 1
        ):
            self.threshold += self.cr
        else:
            pass

    def update_memory(
        self, instances: List, labels: List, weights: Optional[List] = None
    ) -> None:
        """
        Atualiza a matriz de instâncias rotuladas

        Args:
            instances: instâncias
            labels: rotulos
            weights: Pesos de cada classe
        """
        if not weights:
            weights = [1 for _ in range(len(instances))]

        for instance, label, weight in zip(instances, labels, weights):
            self.cl_memory[instance][label] += weight

    def remember(self, X: List) -> List:
        """
        Responsável por armazenar como está as instâncias dado um  momento no
        código

        Args:
            X: Lista com as instâncias

        Returns:
            A lista memorizada em um dado momento
        """
        return [np.argmax(self.cl_memory[x]) for x in X]

    def storage_predict(
        self, idx, confidence, classes
    ) -> Dict[int, Dict[float, int]]:
        """
        Responsável por armazenar o dicionário de dados da matriz

        Args:
            idx: indices de cada instância
            confidence: taxa de confiança para a classe destinada
            classes: indices das classes

        Returns:
            Retorna o dicionário com as classes das instâncias não rotuladas
        """
        memo = {}

        for i, conf, label in zip(idx, confidence, classes):
            memo[i] = {}
            memo[i]["confidence"] = conf
            memo[i]["classes"] = label

        return memo

    def rule_1(self):
        """
        Regra responsável por verificar se as classes são iguais E as duas
            confianças preditas é maior que o limiar

        Returns:
            a lista correspondente pela condição
        """
        selected = []
        classes_selected = []

        for i in self.pred_x_it:
            if (
                self.dict_first[i]["confidence"] >= self.threshold
                and self.pred_x_it[i]["confidence"] >= self.threshold
                and self.dict_first[i]["classes"]
                == self.pred_x_it[i]["classes"]
            ):
                selected.append(i)
                classes_selected.append(self.dict_first[i]["classes"])

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

        for i in self.pred_x_it:
            if (
                self.dict_first[i]["confidence"] >= self.threshold
                or self.pred_x_it[i]["confidence"] >= self.threshold
            ) and self.dict_first[i]["classes"] == self.pred_x_it[i][
                "classes"
            ]:
                selected.append(i)
                classes_selected.append(self.dict_first[i]["classes"])

        return selected, classes_selected

    def rule_3(self):
        """
        regra responsável por verificar se as classes são diferentes E  as
        confianças preditas são maiores que o limiar

        Returns:
        a lista correspondente pela condição
        """
        selected = []

        for i in self.pred_x_it:
            if (
                self.dict_first[i]["classes"] != self.pred_x_it[i]["classes"]
                and self.dict_first[i]["confidence"] >= self.threshold
                and self.pred_x_it[i]["confidence"] >= self.threshold
            ):
                selected.append(i)

        return selected, self.remember(selected)

    def rule_4(self):
        """
        regra responsável por verificar se as classes são diferentes E uma das
        confianças preditas é maior que o limiar

        Returns:
        a lista correspondente pela condição
        """
        selected = []

        for i in self.pred_x_it:
            if self.dict_first[i]["classes"] != self.pred_x_it[i][
                "classes"
            ] and (
                self.dict_first[i]["confidence"] >= self.threshold
                or self.pred_x_it[i]["confidence"] >= self.threshold
            ):
                selected.append(i)

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

        base_estimator_init = clone(self.base_estimator)

        # L0 - MODELO TREINADO E CLASSIFICADO COM L0
        base_estimator_init.fit(
            X[safe_mask(X, has_label)], self.transduction_[has_label]
        )
        # ACC EM L0 - RETORNA A EFICACIA DO MODELO
        init_acc = self.calc_local_measure(
            X[safe_mask(X, self.init_labeled_)],
            y[self.init_labeled_],
            base_estimator_init,
        )
        print(f"Acurácia do novo classificador: {init_acc}")

        return init_acc

    def add_new_labeled(
        self,
        selected_full,
        selected,
        local_acc,
        init_acc,
        max_proba,
        pred,
    ):
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
                f" added {len(selected)} new labels."
            )

    def select_instances_by_rules(self):
        """
        Função responsável por gerenciar todas as regras de inclusão do método

        Returns:
            _type_: _description_
        """
        insertion_rules = [self.rule_1, self.rule_2, self.rule_3, self.rule_4]

        for rule in insertion_rules:
            selected, pred = rule()

            if selected:
                return np.array(selected), pred
        return np.array([]), ""
