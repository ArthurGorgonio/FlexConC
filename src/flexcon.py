import warnings
from statistics import mean

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.utils import safe_mask

from src.utils import validate_estimator


class FlexConC(SelfTrainingClassifier):
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
        self._it = 0
        self._init_acc = 0
        self.verbose = verbose
        self.old_selected = []
        self.selected = []
    
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
        # we need row slicing support for sparce matrices, but costly finiteness check
        # can be delegated to the base estimator.
        
        X, y = self._validate_data(
            X, y, accept_sparse=["csr", "csc", "lil", "dok"], force_all_finite=False
        )

        if self.base_estimator is None:
            raise ValueError("base_estimator cannot be None!")

        self.base_estimator_ = clone(self.base_estimator)

        if self.max_iter is not None and self.max_iter < 0:
            raise ValueError(f"max_iter must be >= 0 or None, got {self.max_iter}")

        if not (0 <= self.threshold < 1):
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

        if np.all(has_label):
            warnings.warn("y contains no unlabeled samples", UserWarning)

        self.transduction_ = np.copy(y)
        self.labeled_iter_ = np.full_like(y, -1)
        self.labeled_iter_[has_label] = 0
        self.init_labeled_ = has_label.copy()

        self.n_iter_ = 0
        #import ipdb; ipdb.sset_trace()
        while not np.all(has_label) and (
            self.max_iter is None or self.n_iter_ < self.max_iter
        ):
            sset_trace()
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

            # Select new labeled samples
            selected = max_proba >= self.threshold
        
            #if()
            
            # Map selected indices into original array
            selected_full = np.nonzero(~has_label)[0][selected]


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
            self.transduction_[selected_full] = pred[selected]
            has_label[selected_full] = True
            self.labeled_iter_[selected_full] = self.n_iter_

            if selected_full.shape[0] > 0:
                #
                # no changed labels
                self.new_threshold(max_proba, len(selected), len(max_proba))
                self.termination_condition_ = "threshold_change"
                self.new_threshold

            if self.verbose:
                print(
                    f"End of iteration {self.n_iter_},"
                    f" added {selected} new labels."
                    # f"\n LOG:\n {has_label}\n"
                )
            else:
                print(
                    f"End of iteration {self.n_iter_},"
                    f" 0 new instaces are labelled."
                )

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
        y_pred = classifier.predict(X)

        return accuracy_score(y_true, y_pred)

    def new_threshold(self, local_measure, init_acc):
        if local_measure > (init_acc + 0.01) and ((self.threshold - self._cr) > 0.0):
            self.threshold -= self._cr
        elif (local_measure < (init_acc - 0.01)) and ((self.threshold + self._cr) <= 1):
            self.threshold += self._cr
