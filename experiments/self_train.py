"""
self_train.py

Self-training com limiar de confiança fixo (mesma ideia do
SelfTrainingClassifier do scikit-learn) e um gancho opcional de
reavaliação de pseudo-rótulos (ex.: DeferredRevaluationStrategy com
SilhouetteMetric).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.semi_supervised import SelfTrainingClassifier


class SelfTraining(SelfTrainingClassifier):
    """
    Parameters
    ----------
    estimator : estimator object
        Classificador base (precisa ter ``predict_proba``, porque usamos a
        probabilidade como medida de confiança).
    revaluation_strategy : RevaluationStrategy, optional
        Objeto com ``revaluate(instances, labels, labeled_mask, **kwargs)``
        que devolve ``(labels, labeled_mask)``. Chamado ao fim de cada
        iteração. Se None, não há reavaliação (self-training puro, que
        serve de baseline para comparar).
    threshold : float, optional
        Confiança mínima para aceitar um pseudo-rótulo.
    max_iter : int, optional
        Número máximo de iterações.
    verbose : int, optional
        0 = silencioso; 1 = resumo; 2 = detalhado por iteração.
    """

    def __init__(
        self,
        estimator,
        revaluation_strategy=None,
        threshold: float = 0.95,
        max_iter: int = 100,
        verbose: int = 5,
    ):
        super().__init__(
            estimator=estimator,
            threshold=threshold,
            criterion='threshold',
            max_iter=max_iter,
            verbose=verbose,
        )
        self.revaluation_strategy = revaluation_strategy
        self.revaluate_counter = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SelfTraining':
        X = np.asarray(X)
        y = np.asarray(y)

        has_label = y != -1

        if self.verbose >= 1:
            print(
                f'[SELF-TRAIN] fit recebeu X={X.shape} y={y.shape} | '
                f'rotulados={int(has_label.sum())} '
                f'sem rótulo={int((~has_label).sum())}'
            )

        self.estimator_ = clone(self.estimator)
        self.transduction_ = np.copy(y)

        self.labeled_iter_ = np.full(y.shape, -1, dtype=np.int64)
        self.labeled_iter_[has_label] = 0

        self.n_iter_ = 0
        self.termination_condition_ = 'max_iter'
        while self.n_iter_ < self.max_iter and not has_label.all():
            self.n_iter_ += 1

            # magia negra?
            estado_antes = (has_label.copy(), self.transduction_.copy())

            # 1) TREINAR só com o que tem rótulo
            self.estimator_.fit(X[has_label], self.transduction_[has_label])

            # 2) PREDIZER nas instâncias sem rótulo
            unlabeled_idx = np.where(~has_label)[0]
            prob = self.estimator_.predict_proba(X[unlabeled_idx])
            pred = self.estimator_.classes_[np.argmax(prob, axis=1)]

            # 3) SELECIONAR por confiança
            selected = prob.max(axis=1) > self.threshold

            if self.verbose >= 2:
                conf = prob.max(axis=1)
                print(
                    f'[SELF-TRAIN] iter {self.n_iter_}: treinou com '
                    f'{int(has_label.sum())} rotuladas | '
                    f'{len(unlabeled_idx)} sem rótulo | '
                    f'confiança máx={conf.max():.3f} média={conf.mean():.3f} | '
                    f'acima de {self.threshold}: {int(selected.sum())}'
                )

            if not selected.any():
                print('Select 0 instances, doing revaluation...\n\n\n')
                if self.revaluation_strategy:
                    print('Revaluation...\n\n\n')
                    revaluate = False
                    n_antes = int(has_label.sum())
                    self.transduction_, has_label = (
                        self.revaluation_strategy.revaluate(
                            X,
                            self.transduction_,
                            has_label,
                            estimator=self.estimator_,
                            labeled_iter=self.labeled_iter_,
                        )
                    )
                    if has_label.any():
                        self.revaluate_counter += 1
                        self.labeled_iter_[~has_label] = -1
                        revaluate = True

                    if self.verbose >= 2:
                        n_depois = int(has_label.sum())
                        print(
                            f'[SELF-TRAIN] reavaliação: {n_antes} -> {n_depois} '
                            f'rotuladas (devolvidas ao pool: {n_antes - n_depois})'
                        )

                    if not revaluate:
                        self.termination_condition_ = 'no_change'
                        print('Não quis fazer a reavaliação, tô cansado, flw!')
                        break

            # 4) ROTULAR
            chosen = unlabeled_idx[selected]
            self.transduction_[chosen] = pred[selected]
            self.labeled_iter_[chosen] = self.n_iter_
            has_label[chosen] = True

            if self.verbose >= 2:
                print(
                    f'[SELF-TRAIN] iter {self.n_iter_}: aceitou +{len(chosen)} '
                    f'pseudo-rótulos | ficam {int((~has_label).sum())} sem rótulo'
                )

            # 5) REAVALIAR (opcional)

            if np.array_equal(has_label, estado_antes[0]) and np.array_equal(
                self.transduction_, estado_antes[1]
            ):
                self.termination_condition_ = 'no_change'
                break

        if has_label.all():
            self.termination_condition_ = 'all_labeled'


        print(f'Reavaliação {self.revaluate_counter}')
        # Treino FINAL com o conjunto de rótulos consolidado
        self.estimator_.fit(X[has_label], self.transduction_[has_label])
        self.classes_ = self.estimator_.classes_

        if self.verbose >= 1:
            it, cnt = np.unique(self.labeled_iter_, return_counts=True)
            print(
                f'[SELF-TRAIN] fim: parada={self.termination_condition_} '
                f'n_iter={self.n_iter_} | por iteração de entrada '
                f'(-1=sem rótulo, 0=real): {dict(zip(it.tolist(), cnt.tolist()))}'
            )

        return self
