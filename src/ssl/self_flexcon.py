from typing import Any, Dict

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import safe_mask

from src.ssl.flexcon import BaseFlexConC


class SelfFlexCon(BaseFlexConC):
    """
    Implementação do FlexConC no algoritmo Self-training.

    Parameters
    ----------
        params : Dict[str, Any]
            Parâmetros que são passados para a superclasse para ela
            realizar as setagens, por default None. Se não receber um
            classificador no atributo `base_estimator` será utilizado
            um Naïve Bayes como classificador do método.
    """
    def __init__(self, params: Dict[str, Any] = None):
        if params is None or "base_estimator" not in params.keys():
            params = {
                "base_estimator": GaussianNB()
            }
        super().__init__(**params)
        self.n_iter_ = 0

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
            raise ValueError("y contains no unlabeled sample")

        init_acc = self.train_new_classifier(has_label, X, y)
        old_selected = []

        while not np.all(has_label) and (
            self.max_iter is None or self.n_iter_ <= self.max_iter
        ):
            self.n_iter_ += 1
            self.base_estimator_.fit(
                X[safe_mask(X, has_label)], self.transduction_[has_label]
            )

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
                # WIP - transformar o selected num vetor bool do tamanho da predição (pred)
                # Assim, é possível fazer as operações
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
            self.add_new_labeled(selected_full, selected, pred)
            self.update_memory(np.nonzero(~has_label)[0], pred[selected])
            # Predict on the labeled samples
            try:
                if old_selected:
                    selected_full = np.array(
                        old_selected + selected_full.tolist()
                    )
                    new_pred = np.concatenate(
                        (self.transduction_[old_selected], pred[selected])
                    )
                    self.base_estimator_select_.fit(X[selected_full], new_pred)
                    old_selected = []
                else:
                    # Traning model to classify the labeled samples
                    self.base_estimator_select_.fit(
                        X[selected_full], pred[selected]
                    )

                local_acc = self.calc_local_measure(
                    X[safe_mask(X, self.init_labeled_)],
                    y[self.init_labeled_],
                    self.base_estimator_select_,
                )
                self.new_threshold(local_acc, init_acc)
            except ValueError:
                old_selected = selected_full.tolist()

        if self.n_iter_ == self.max_iter:
            self.termination_condition_ = "max_iter"

        if np.all(has_label):
            self.termination_condition_ = "all_labeled"

        self.base_estimator_.fit(
            X[safe_mask(X, has_label)], self.transduction_[has_label]
        )
        self.classes_ = self.base_estimator_.classes_

        return self
