from unittest import TestCase

from mock import Mock, patch

from src.core.core import Core
from src.ssl.self_flexcon import SelfFlexCon


class EnsembleMock(Mock):
    def add_classifier(self, classifier, need_train):
        ...


class DetectorMock(Mock):
    def detector(self, params_detector):
        ...


class ReactorMock(Mock):
    def reactor(self, params_reactor):
        ...


class MetricsMock(Mock):
    def accuracy_score(self, y_true, y_pred):
        return 1.0

    def f1_score(self, y_true, y_pred):
        return 1.0


class TestCore(TestCase):
    def setUp(self):
        self.core = Core(EnsembleMock, DetectorMock, ReactorMock)

    def test_init_should_return_valid_class_instance_when_args_are_valid(self):
        self.assertEqual(
            [
                self.core.ensemble,
                self.core.detect,
                self.core.react,
                self.core.chunk_size,
                self.core.metrics,
                self.core.metrics_calls
            ],
            [
                EnsembleMock,
                DetectorMock,
                ReactorMock,
                500,
                {},
                {}
            ]
        )

    def test_configure_param_should_return_valid_configurations_when_args_are_valid(  # NOQA
        self
    ):
        expected_output = {
            'ssl': {
                'cr': 0.05,
                'threshold': 0.95
            },
            'detector': {
                'threshold': 0.2
            },
            'reaction': {
                'threshold': 0.5
            }
        }

        ssl_params = {
            'cr': 0.05,
            'threshold': 0.95
        }
        detector_params = {
            'threshold': 0.2
        }
        reactor_params = {
            'thr': 0.5
        }
        core = Core()
        core.configure_params(
            SelfFlexCon,
            ssl_params,
            detector_params,
            reactor_params
        )
        self.assertEqual(
            {
                'ssl': {
                    'cr': core.ensemble.ssl_params["cr"],
                    'threshold': core.ensemble.ssl_params["threshold"]
                },
                'detector': {
                    'threshold': core.detect.detection_threshold
                },
                'reaction': {
                    'threshold': core.react.thr
                }
            },
            expected_output
        )

    @patch("src.core.core.Ensemble")
    def test_configurae_ensemble_should_create_ensemble_when_all_in_configured_with_four_classifiers(  # NOQA
        self,
        ensemble
    ):
        base_classifiers = [1, 2, 3, 4]
        self.core = Core(ensemble)
        self.core.configure_params(ensemble, {}, {}, {})
        self.core.configure_classifier(base_classifiers)

        self.assertEqual(self.core.ensemble.add_classifier.call_count, 4)

    def test_add_metrics_should_return_dict_with_metric_callable(self):
        metrics = MetricsMock()
        self.core.add_metrics("acc", metrics.accuracy_score)
        self.core.add_metrics("f1", metrics.f1_score)

        self.assertEqual(
            self.core.metrics_calls,
            {
                "acc": metrics.accuracy_score,
                "f1": metrics.f1_score
            }
        )

        self.assertEqual(
            self.core.metrics,
            {
                "acc": [],
                "f1": []
            }
        )

    def test_evaluate_metrics_should_return_two_values_when_two_metrics_are_used(self):  # NOQA
        evaluate = MetricsMock()
        self.core.add_metrics("acc", evaluate.accuracy_score)
        self.core.add_metrics("f1", evaluate.f1_score)

        y_true = [1, 1, 0, 0, 0, 1]
        y_pred = [1, 1, 0, 0, 0, 1]

        self.core.evaluate_metrics(y_true, y_pred)

        self.assertEqual(self.core.metrics, {'acc': [1.0], 'f1': [1.0]})
