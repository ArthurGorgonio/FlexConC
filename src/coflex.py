import warnings
from random import shuffle
from statistics import mode

import numpy as np
from sklearn.base import clone
from sklearn.utils import safe_mask

from src.flexcon import FlexConC
from src.utils.utils import validate_estimator


class CoFlex(FlexConC):
    """
    Classe responsável por instanciar o FlexConC
    """

    def __init__(self, base_estimator) -> None:
        super().__init__(base_estimator)
        self.base_estimator_second_ = clone(base_estimator)
        self.base_estimator_select_second_ = clone(base_estimator)
    
    def fit(self, X, y):
        """
        Fit co-training classifier using `X`, `y` as training data.
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
        
        X1, X2 = self.split_data(X)
        y2 = np.copy(y)

        if y.dtype.kind in ["U", "S"]:
            raise ValueError(
                "y has dtype string. If you wish to predict on "
                "string targets, use dtype object, and use -1"
                " as the label for unlabeled samples."
            )

        has_label = y != -1
        has_label2 = y != -1
        self.cl_memory = [[0] * np.unique(y[has_label]) for _ in range(len(X))]

        if np.all(has_label):
            warnings.warn("y contains no unlabeled samples", UserWarning)

        init_acc = self.train_new_classifier(has_label, X, y)
        old_selected = []
        old_selected2 = []
        old_pred = []
        old_pred2 = []
        self.n_iter_ = 0

        while not np.all(has_label) and (
            self.max_iter is None or self.n_iter_ <= self.max_iter
        ):
            self.n_iter_ += 1
            self.base_estimator_.fit(
                X1[safe_mask(X1, has_label)], self.transduction_[has_label]
            )

            self.base_estimator_second_.fit(
                X2[safe_mask(X2, has_label2)], self.transduction_[has_label2]
            )
            

            # Validate the fitted estimator since `predict_proba` can be
            # delegated to an underlying "final" fitted estimator as
            # generally done in meta-estimator or pipeline.
            validate_estimator(self.base_estimator_)

            # Predict on the unlabeled samples - VIEW 1
            prob1 = self.base_estimator_.predict_proba(
                X1[safe_mask(X1, ~has_label)]
            )
            pred1 = self.base_estimator_.classes_[np.argmax(prob1, axis=1)]
            max_proba1 = np.max(prob1, axis=1)

            # Predict on the unlabeled samples - VIEW 2
            prob2 = self.base_estimator_second_.predict_proba(
                X2[safe_mask(X2, ~has_label2)]
            )
            pred2 = self.base_estimator_second_.classes_[np.argmax(prob2, axis=1)]
            max_proba2 = np.max(prob2, axis=1)

            if self.n_iter_ > 1:
                self.pred_x_it = self.storage_predict(
                    idx=np.nonzero(~has_label)[0],
                    confidence=max_proba1,
                    classes=pred1,
                )

                self.pred_x_it = self.storage_predict(
                    idx=np.nonzero(~has_label)[0],
                    confidence=max_proba2,
                    classes=pred2,
                )

                pseudo_ids = np.nonzero(~has_label)[0].tolist()
                pseudo_ids2 = np.nonzero(~has_label2)[0].tolist()
                selected_full, pred_full = self.select_instances_by_rules()
                selected_full2, pred_full2 = self.select_instances_by_rules()

                if not selected_full:
                    self.threshold = np.max(max_proba1)
                    self.threshold2 = np.max(max_proba2)
                    selected_full, pred_full = self.select_instances_by_rules()
                    selected_full2, pred_full2 = self.select_instances_by_rules()
                selected = [pseudo_ids.index(inst) for inst in selected_full]
                selected2 = [pseudo_ids2.index(inst) for inst in selected_full2]
                pred1[selected] = pred_full
                pred2[selected2] = pred_full2
            else:
                self.dict_first = self.storage_predict(
                    idx=np.nonzero(~has_label)[0],
                    confidence=max_proba1,
                    classes=pred1,
                )
                self.dict_first2 = self.storage_predict(
                    idx=np.nonzero(~has_label2)[0],
                    confidence=max_proba2,
                    classes=pred2,
                )
                # Select new labeled samples
                selected = max_proba1 >= self.threshold
                selected2 = max_proba2 >= self.threshold
                # Map selected indices into original array
                selected_full = np.nonzero(~has_label)[0][selected]
                selected_full2 = np.nonzero(~has_label2)[0][selected2]

            # Add newly labeled confident predictions to the dataset
            has_label[selected_full] = True
            has_label2[selected_full2] = True

            self.update_memory(np.nonzero(~has_label)[0], pred1)
            self.update_memory(np.nonzero(~has_label2)[0], pred2)
            # Predict on the labeled samples
            try:
                # if old_selected and old_selected2:
                #     selected_full = np.array(old_selected + selected_full.tolist())
                #     selected_full2 = np.array(old_selected2 + selected_full2.tolist())
                #     selected = old_pred + selected
                #     selected2 = old_pred2 + selected2
                #     old_selected = []
                #     old_selected2 = []
                #     old_pred = []
                #     old_pred2 = []

                # Traning model to classify the labeled samples
                self.base_estimator_select_.fit(
                    X1[selected_full], pred1[selected]
                )
                # Traning model to classify the labeled samples
                self.base_estimator_select_second_.fit(
                    X2[selected_full2], pred2[selected2]
                )

                local_acc = self.calc_local_measure(
                    X1[safe_mask(X1, self.init_labeled_)],
                    y[self.init_labeled_],
                    self.base_estimator_select_,
                )
                local_acc2 = self.calc_local_measure(
                    X2[safe_mask(X2, self.init_labeled_)],
                    y2[self.init_labeled_],
                    self.base_estimator_select_second_,
                )
                self.add_new_labeled(
                    selected_full,
                    selected,
                    pred2,
                )
                self.add_new_labeled(
                    selected_full2,
                    selected2,
                    pred1,
                )
            except ValueError:
                old_selected = selected_full.tolist()
                old_pred = pred1[selected].tolist()
                old_selected2 = selected_full2.tolist()
                old_pred2 = pred2[selected2].tolist()

        if self.n_iter_ == self.max_iter:
            self.termination_condition_ = "max_iter"

        if np.all(has_label):
            self.termination_condition_ = "all_labeled"

        self.base_estimator_.fit(
            X[safe_mask(X, has_label)], self.transduction_[has_label]
        )
        self.classes_ = self.base_estimator_.classes_

        return self
    
    def split_data(self, X):
        features = [i for i in range(len(X[0]))]
        shuffle(features)
        return X[:, :len(features) // 2], X[:, len(features) // 2:]
