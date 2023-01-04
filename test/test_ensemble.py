from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np

from src.ssl.ensemble import Ensemble


class ClassifierMock(Mock):
    def predict(self, instances):
        ...

    def fit(self, instances, labels):
        ...

    def __str__(self) -> str:
        msg = f'Nome do classificador é: {self.cl_name}'
        return msg


class FlexConMock(Mock):
    ...


class TestEnsemble(TestCase):
    def setUp(self) -> None:
        self.ensemble = Ensemble(ssl_algorithm=FlexConMock())

    def test_add_classifier_should_return_ssl_classifier_when_more_than_one_classier_is_added(self):  # NOQA
        base_classifier = ["cl_1", "cl_2", "cl_3", "cl_4", "cl_5"]

        for cl in base_classifier:
            self.ensemble.ssl_algorithm.return_value = cl
            self.ensemble.add_classifier(cl)

        self.assertEqual(
            self.ensemble.ssl_algorithm.call_count,
            len(base_classifier)
        )
        self.assertListEqual(self.ensemble.ensemble, base_classifier)

    def test_add_classifier_should_return_empty_ensemble_when_no_classifier_is_added(self):  # NOQA
        base_classifier = []

        for cl in base_classifier:
            self.ensemble.ssl_algorithm.return_value = cl
            self.ensemble.add_classifier(cl)

        self.assertEqual(
            self.ensemble.ssl_algorithm.call_count,
            len(base_classifier)
        )
        self.assertListEqual(self.ensemble.ensemble, base_classifier)

    def test_remove_classifier_should_remove_a_classifier_when_it_exists(self):  # NOQA
        base_classifiers = ["cl_1", "cl_2"]

        for cl in base_classifiers:
            self.ensemble.ssl_algorithm.return_value = cl
            self.ensemble.add_classifier(cl)

        self.ensemble.remover_classifier("cl_2")

        self.assertListEqual(self.ensemble.ensemble, ["cl_1"])

    def test_remove_classifier_should_empty_ensemble_when_it_not_exists(self):  # NOQA
        base_classifiers = ["cl_1", "cl_2"]

        for cl in base_classifiers:
            self.ensemble.ssl_algorithm.return_value = cl
            self.ensemble.add_classifier(cl)

        with self.assertRaises(ValueError):
            self.ensemble.remover_classifier("cl_1")
            self.ensemble.remover_classifier("cl_2")
            self.ensemble.remover_classifier("cl_3")

        self.assertListEqual(self.ensemble.ensemble, [])

    @patch("src.ssl.ensemble.Ensemble.predict_one_classifier")
    @patch("src.ssl.ensemble.accuracy_score")
    def test_measure_ensemble_should_return_base_classifier_(self, accuracy_score, predict):  # NOQA
        accuracy_score.return_value = 1
        base_classifier = [ClassifierMock, ClassifierMock]

        for cl in base_classifier:
            self.ensemble.ssl_algorithm.return_value = cl()
            self.ensemble.add_classifier(cl)

        acc = self.ensemble.measure_ensemble(instances=[], classes=[])

        self.assertListEqual(acc, [1, 1])
        self.assertEqual(predict.call_count, 2)

    def test_drop_ensemble_should_return_empty_list(self):
        base_classifier = [ClassifierMock, ClassifierMock]

        for cl in base_classifier:
            self.ensemble.ssl_algorithm.return_value = cl()
            self.ensemble.add_classifier(cl)

        self.ensemble.drop_ensemble()

        self.assertListEqual(self.ensemble.ensemble, [])

    @patch("src.ssl.ensemble.Ensemble.fit_single_classifier")
    def test_fit_ensemble_should_return_fitted_ensemble_with_three_members(
        self,
        fit
    ):
        base_classifier = [ClassifierMock, ClassifierMock, ClassifierMock]

        for cl in base_classifier:
            self.ensemble.ssl_algorithm.return_value = cl()
            self.ensemble.add_classifier(cl)

        self.ensemble.fit_ensemble([], [])

        self.assertEqual(fit.call_count, 3)

    def test_swap_ensemble_should_return_swaped_ensemble(self):
        base_classifier = ["cl_1", "cl_2", "cl_3", "cl_4"]
        exchanged_classifiers = ["cl_5", "cl_6", "cl_7", "cl_8"]

        for cl in base_classifier:
            self.ensemble.ssl_algorithm.return_value = cl
            self.ensemble.add_classifier(cl)

        self.ensemble.swap(exchanged_classifiers, [3, 1, 2, 0], [], [], False)

        self.assertListEqual(
            self.ensemble.ensemble,
            ["cl_8", "cl_6", "cl_7", "cl_5"]
        )

    def test_pareto_frontier(self):
        expected_minimization_output = [
            (2, 3),  # A
            (1, 5),  # B
            (9, 1),  # C
            (9, 2),  # F
        ]
        expected_maximization_output = [
            (9, 1),  # C
            (9, 2),  # F
            (9, 4),  # G
            (6, 6),  # H
            (5, 7),  # I
            (4, 8),  # J
            (3, 9),  # K
        ]

        pts = [
            (2, 3),  # A
            (1, 5),  # B
            (9, 1),  # C
            (4, 6),  # D
            (3, 7),  # E
            (9, 2),  # F
            (9, 4),  # G
            (6, 6),  # H
            (5, 7),  # I
            (4, 8),  # J
            (3, 9),  # K
        ]

        # pts = [
        #     (.2, .3),
        #     (.1, .5),
        #     (.9, .1),
        #     (.4, .6),
        #     (.3, .7),
        #     (.9, .2),
        #     (.9, .4),
        #     (.6, .6),
        #     (.5, .7),
        #     (.4, .8),
        #     (.3, .9)
        # ]

        self.assertEqual(
            self.ensemble.pareto_frontier(pts, True),
            expected_minimization_output
        )

        self.assertEqual(
            self.ensemble.pareto_frontier(pts),
            expected_maximization_output
        )

    @patch('src.ssl.ensemble.Ensemble.q_measure_classifier_vs_ensemble')
    @patch('src.ssl.ensemble.Ensemble.measure_ensemble')
    def test_map_ensemble_pareto(self, acc, q_measure):
        acc_result = [.2, .1, .9, .4, .3, .12, .9, .6, .5, .4, .3]

        for i in range(len(acc_result)):
            self.ensemble.add_classifier(ClassifierMock(), False)
            self.ensemble.ensemble[-1].name = f'cl_{i}'

        expected_classifier = ['cl_2', 'cl_6']

        acc.return_value = acc_result
        q_measure.return_value = np.array(
            [.3, .5, .1, .6, .7, .2, .4, .6, .7, .8, .9]
        )

        self.ensemble.compute_pareto_frontier([], [], '')

        ensemble_names = [cl.name for cl in self.ensemble.ensemble]

        self.assertEqual(
            ensemble_names,
            expected_classifier
        )

    def test_similarity_calculation(self,):
        ensemble = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1])
        classifier = np.array([0, 1, 0, 1, 1, 0, 0, 1, 1, 0])

        expected_similarity = 0.0

        self.assertEqual(
            self.ensemble._evaluate_similarity(ensemble, classifier),
            expected_similarity
        )

    # @patch("src.ssl.ensemble.compare_labels")
    @patch("src.ssl.ensemble.Ensemble.predict_one_classifier")
    @patch("src.ssl.ensemble.Ensemble.predict_ensemble")
    def test_q_similarity_calculation(self, ensemble, classifiers):
        classes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ensemble.return_value = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1])
        classifiers.return_value = np.array([1, 1, 0, 1, 1, 0, 0, 1, 1, 0])

        expected_output = [-.3333333333333333, -.3333333333333333]

        for _ in range(2):
            self.ensemble.add_classifier(ClassifierMock(), False)

        self.assertListEqual(
            self.ensemble.q_measure_classifier_vs_ensemble(
                [],
                classes
            ).tolist(),
            expected_output
        )

    @patch("src.ssl.ensemble.Ensemble.predict_one_classifier")
    @patch("src.ssl.ensemble.Ensemble.predict_ensemble")
    def test_q_similarity_calculation_without_classifier_in_ensemble(
        self,
        ensemble,
        classifiers
    ):
        classes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ensemble.return_value = []
        classifiers.return_value = []

        expected_output = []

        self.assertEqual(
            self.ensemble.q_measure_classifier_vs_ensemble(
                [],
                classes
            ).tolist(),
            expected_output
        )
