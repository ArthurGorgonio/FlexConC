from unittest import TestCase
from unittest.mock import Mock, patch

from src.ssl.ensemble import Ensemble


class ClassifierMock(Mock):
    def predict(self, instances):
        ...

    def fit(self, instances, labels):
        ...


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
