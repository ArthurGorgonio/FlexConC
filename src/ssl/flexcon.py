from abc import abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.utils import safe_mask

from src.utils import validate_estimator


class BaseFlexConC(SelfTrainingClassifier):
    """
    Classe base do método Flexible Conficende with Classifier que
    contém os métodos genéricos. Esse é um método semissupervisionado
    utilizado na tarefa de classificação de dados.

    Parameters
    ----------
        base_estimator : estimator object
            Classificador que irá ser treinado.
        cr : float, optional
            Taxa de mudança do limiar `threshold`, por default 0.05.
        threshold : float, optional
            Limiar do método. Esse parâmetro limita quais instâncias
            serão selecionadas para a etapa de classificação do método,
            por default 0.95. Adicionalmente, esse limiar deve assumir
            valores entre 0.0 e 1.0
        verbose : bool, optional
            Loga alguns dados importantes do método para a tela, por
            default False.
    """

    def __init__(self, base_estimator, cr=0.05, threshold=0.95, verbose=False):
        super().__init__(
            base_estimator=base_estimator, threshold=threshold, max_iter=100
        )
        self.validate()
        self.cr: float = cr
        self.verbose = verbose
        self.old_selected: List = []
        self.pred_1_it: Dict = {}
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

    @abstractmethod
    def fit(self, X, y):
        ...

    def validate(self):
        # Validate the fitted estimator since `predict_proba` can be
        # delegated to an underlying "final" fitted estimator as
        # generally done in meta-estimator or pipeline.
        try:
            validate_estimator(self.base_estimator)
        except ValueError:
            return False
        return True

    def calc_local_measure(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        classifier
    ) -> float:
        """
        Calcula a eficácia de classificação de um modelo que foi
        treinado com as instâncias atuais. Esse modelo será avaliado
        com as instâncias inicialmente rotuladas para validar sua
        generalização.

        Parameters
        ----------
        X : np.ndarray
            Instâncias inicialmente rotuladas.
        y_true : np.ndarray
            Rótulo das instâncias.
        classifier : estimator object
            Classificador que irá predizer as instâncias.

        Returns
        -------
        float
            Valor da acurácia do modelo, mensurado nas instâncias
            inicialmente rotuladas.
        """
        y_pred = classifier.predict(X)

        return accuracy_score(y_true, y_pred)

    def new_threshold(self, local_measure: float, init_acc: float) -> None:
        """
        Calcula o novo `BaseFlexConC.threshold` com base em alguns
        critérios:
            - local_measure > init_acc + δ => threshold -= cr
                - Modelo está generalizando bem, pode reduzir o nível
                de restrição para incluir novas instâncias.
            - local_measure < init_acc - δ => threshold += cr
                - Modelo não está generalizando bem, aumentando o nível
                de restrição para incluir novas instâncias.
            - init_acc - δ <= local_measure <= init_acc + δ => threshold
                - Variação aceitável da métrica de eficácia. Não é
                preciso alterar o threshold.

        Parameters
        ----------
        local_measure : float
            Valor da acurácia atual, ou seja, acurácia do modelo quando
            treinado com as instâncias selecionadas e predizendo as
            instâncias inicialmente rotuladas.
        init_acc : float
            Valor da acurácia inicial, ou seja, acurácia do modelo nas
            instâncias inicialmente rotuladas.
        """
        if (
            local_measure > init_acc + 0.01
            and self.threshold - self.cr >= 0.0
        ):
            self.threshold -= self.cr
        elif (
            local_measure < init_acc - 0.01
            and self.threshold + self.cr <= 1.0
        ):
            self.threshold += self.cr
        else:
            pass

    def update_memory(
        self,
        instances: List,
        labels: List[int],
        weights: Optional[List[float]] = None
    ) -> None:
        """
        Atualiza a memória de classificação que é utilizada para
        recuperar o rótulo da instância para as regras de inclusão
        `rule3` e `rule4`.

        Parameters
        ----------
        instances : List[int]
            Posições das instâncias que serão atualizadas.
        labels : List[float]
            Rótulos das instâncias.
        weights : Optional[List], optional
            Pessos. Esse parâmetro deve ser passado caso não esteja
            utilizando o combinador votação simples no comitê. Por
            padrão esse atributo é None, dessa forma serão compudados
            pesos uniformes para cada classificador
        """

        if not weights:
            weights = [1 for _ in range(len(instances))]

        for instance, label, weight in zip(instances, labels, weights):
            self.cl_memory[instance][label] += weight

    def remember(self, X: List[int]) -> List[int]:
        """
        Informa qual foi a classe cuja instância foi mais vezes
        rotulada.

        Parameters
        ----------
        X : List[int]
            Lista das instâncias que serão analisadas para informar em
            qual classe a respectiva instâncias foi mais vezes
            classificadas.

        Returns
        -------
        List[int]
            Lista com os rótulos das instâncias.
        """
        return [np.argmax(self.cl_memory[x]) for x in X]

    def storage_predict(
        self,
        idx: List[int],
        confidence: List[float],
        classes: List[int],
    ) -> Dict[int, Dict[float, int]]:
        """
        Cria uma estrutura que armazena a instância, confiança e o
        rótulo que o modelo predisse. Essa estrutura é guardada em um
        dicionário dentro de outro dicionário, conforme exemplo:

        ```python
        memory = {
            1: {
                'confidence': 0.952,
                'classes': 0
            },
            2: {
                'confidence': 0.45,
                'classes': 1
            },
        }
        ```

        Parameters
        ----------
        idx : List[int]
            Lista com os indices das instância.
        confidence : List[float]
            Lista com as taxa de confiança (confiabilidade) da predição
            do classificador para uma determinada classe.
        classes : List[int]
            Classe que foi atribuída a instância pelo modelo preditivo.

        Returns
        -------
        Dict[int, Dict[float, int]]
            Retorna um estrutura onde pode recuperar as predições e
            valor de confiança da predição de uma instância específica.
        """
        memo = {}

        for i, conf, label in zip(idx, confidence, classes):
            memo[i] = {
                "confidence": conf,
                "classes": label,
            }

        return memo

    def rule_1(self) -> tuple[List[int], List[int]]:
        """
        Seleciona todas as instâncias cuja a classe diverge entre as
        predições E uma das taxas de confiança é maior que o valor do
        limiar ~`BaseFlexConC.threshold` atual. A decisão do rótulo é
        dada pela própria classe da instância, visto que é a mesma
        entre as predições.

        Returns
        -------
        tuple[List[int], List[int]]
            O primeiro elemento da tupla é uma lista indicando as
            instâncias que foram selecionadas por essa regra de
            inserção. Por sua vez, o segundo parâmero da tupla indica
            os rótulos das instâncias.
        """
        selected = []
        classes_selected = []

        for i in self.pred_x_it:
            if (
                self.pred_1_it[i]["classes"] == self.pred_x_it[i]["classes"]
                and (
                    self.pred_1_it[i]["confidence"] >= self.threshold
                    and self.pred_x_it[i]["confidence"] >= self.threshold
                )
            ):
                selected.append(i)
                classes_selected.append(self.pred_1_it[i]["classes"])

        return selected, classes_selected

    def rule_2(self) -> tuple[List[int], List[int]]:
        """
        Seleciona todas as instâncias cuja a classe são iguais entre as
        predições E uma das taxas de confiança é maior que o valor do
        limiar ~`BaseFlexConC.threshold` atual. A decisão do rótulo é
        dada pela própria classe da instância, visto que é a mesma
        entre as predições.

        Returns
        -------
        tuple[List[int], List[int]]
            O primeiro elemento da tupla é uma lista indicando as
            instâncias que foram selecionadas por essa regra de
            inserção. Por sua vez, o segundo parâmero da tupla indica
            os rótulos das instâncias.
        """

        selected = []
        classes_selected = []

        for i in self.pred_x_it:
            if (
                self.pred_1_it[i]["classes"] == self.pred_x_it[i]["classes"]
                and (
                    self.pred_1_it[i]["confidence"] >= self.threshold
                    or self.pred_x_it[i]["confidence"] >= self.threshold
                )
            ):
                selected.append(i)
                classes_selected.append(self.pred_1_it[i]["classes"])

        return selected, classes_selected

    def rule_3(self) -> tuple[List[int], List[int]]:
        """
        Seleciona todas as instâncias cuja a classe diverge entre as
        predições E ambas taxas de confiança são maiores que o valor do
        limiar ~`BaseFlexConC.threshold` atual. A decisão do rótulo é
        dada pela memória de classificação `cl_memory` das instâncias
        ~`BaseFlexConC.remember()`.

        Returns
        -------
        tuple[List[int], List[int]]
            O primeiro elemento da tupla é uma lista indicando as
            instâncias que foram selecionadas por essa regra de
            inserção. Por sua vez, o segundo parâmero da tupla indica
            os rótulos das instâncias.
        """
        selected = []

        for i in self.pred_x_it:
            if (
                self.pred_1_it[i]["classes"] != self.pred_x_it[i]["classes"]
                and (
                    self.pred_1_it[i]["confidence"] >= self.threshold
                    and self.pred_x_it[i]["confidence"] >= self.threshold
                )
            ):
                selected.append(i)

        return selected, self.remember(selected)

    def rule_4(self) -> tuple[List[int], List[int]]:
        """
        Seleciona todas as instâncias cuja a classe diverge entre as
        predições E uma das taxas de confiança é maior que o valor do
        limiar ~`BaseFlexConC.threshold` atual. A decisão do rótulo é
        dada pela memória de classificação `cl_memory` das instâncias
        ~`BaseFlexConC.remember()`.

        Returns
        -------
        tuple[List[int], List[int]]
            O primeiro elemento da tupla é uma lista indicando as
            instâncias que foram selecionadas por essa regra de
            inserção. Por sua vez, o segundo parâmero da tupla indica
            os rótulos das instâncias.
        """
        selected = []

        for i in self.pred_x_it:
            if (
                self.pred_1_it[i]["classes"] != self.pred_x_it[i]["classes"]
                and (
                    self.pred_1_it[i]["confidence"] >= self.threshold
                    or self.pred_x_it[i]["confidence"] >= self.threshold
                )
            ):
                selected.append(i)

        return selected, self.remember(selected)

    def train_new_classifier(
        self,
        has_label: List[bool],
        X: np.ndarray,
        y: np.ndarray
    ) -> float:
        """
        Responsável por treinar um classificador e mensurar
            a sua acertividade.

        Parameters
        ----------
        has_label : List[bool]
            lista com as instâncias rotuladas
        X : ndarray
            Instâncias a serem utilizadas no treinamento e teste do
            classificador.
        y : ndarray
            Rótulos das instâncias.

        Returns
        -------
        float
            Acurácia do classificador nas instâncias inicialmente
            rotuladas da base de dados.
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

        return init_acc

    def add_new_labeled(
        self,
        selected_full: List[int],
        selected: List[float],
        pred: List[int]
    ) -> None:
        """
        Função que atribui as intâncias rotuladas as variáveis de
        controle.

        Parameters
        ----------
        selected_full : List[int]
            lista com os indices das instâncias originais.
        selected : List[float]
            lista das intâncias com acc acima do limiar.
        pred : List[int]
            predição das instâncias não rotuladas.
        """
        self.transduction_[selected_full] = pred[selected]
        self.labeled_iter_[selected_full] = self.n_iter_

    def select_instances_by_rules(
        self
    ) -> tuple[np.ndarray, Union[List[int], str]]:
        """
        Função responsável por executa as chamadas de todas as regras
        de inclusão do método.

        Returns
        -------
        tupla(ndarray, list[int])
            O primeiro elemento da tupla é um array indicando as
            instâncias que serão selecionadas pelas regras de inserção
            existentes no método. O segundo parâmero da tupla indica os
            rótulos das instâncias (se não houver instâncias será
            retornado uma string vazia).
        """
        insertion_rules = [self.rule_1, self.rule_2, self.rule_3, self.rule_4]

        for rule in insertion_rules:
            selected, pred = rule()

            if selected:
                return np.array(selected), pred
        return np.array([]), ""
