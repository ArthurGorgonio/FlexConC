from mock import Mock, patch
from unittest import TestCase

from src.core.core import Core


class EnsembleMock(Mock):
    def add_classifier(self, cla):
        ...

class TestCore(TestCase):
    def setUp(self):
        self.core = Core()
    
    def test_should_return_error_when_configure_classifier_is_called_without_ensemble(self):  # NOQA
        base_classifiers = [1,2,3,4]
        with self.assertRaises(ValueError):
            self.core.configure_classifier(base_classifiers)
    
    @patch("src.core.core.Ensemble")
    def test_should_create_ensenble_when_all_in_configured(self, ensemble):
        ensemble.return_value = EnsembleMock()
        base_classifiers = [1,2,3,4]
        self.core.configure_params("", "", "")
        self.core.configure_classifier(base_classifiers)

        self.assertCountEqual(ensemble.add_classifier(), 4)