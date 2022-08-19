from unittest import TestCase
from unittest.mock import patch

from src.utils import Log


class LoggerTest(TestCase):
    def setUp(self) -> None:
        self.logger = Log()
    
    def test_singleton_creation(self):
        self.logger.filename = 'testes.txt'

        self.assertEqual(self.logger.filename, Log().filename)
        self.assertEqual(self.logger, Log())

    @patch('src.utils.Log.write_archive')
    def test_should_write_in_archive_header_log(self, archive):
        self.logger.write_archive_header()

        archive.called_once()

    @patch('src.utils.Log.write_archive')
    def test_should_write_in_archive_log_of_one_iteration(self, archive):
        iteration_info = {
            'version': '<class "mock.mock.DetectorMock">',
            'ensemble_size': 10,
            'ensemble_hits': 100,
            'drift_detected': False,
            'instances': 400,
            'enlapsed_time': 3213.321,
            'metrics': {
                'acc': 0.99,
                'f1': 0.89,
                'kappa': 0.90,
            },
        }
        self.logger.write_archive_output(**iteration_info)

        archive.called_once()
