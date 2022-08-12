from unittest import TestCase

from mock import Mock, patch

from src.detection.fixed_threshold import FixedThreshold


class TestFixedThreshold(TestCase):
    def setUp(self) -> None:
        self.detector = FixedThreshold()

    def test_creation_class_should_return_valid_class_with_thr_eigty_percent(
        self
    ):
        self.assertEqual(self.detector.detection_threshold, 0.8)

    def test_update_chunks_should_update_a_chunk_with_new_instances(self):
        chunk_1_it = [1, 2, 3, 4, 5]
        chunk_2_it = [6, 7, 8, 9, 10]

        self.assertEqual(self.detector.last_chunk, None)
        self.assertEqual(self.detector.actual_chunk, None)

        self.detector.update_chunks(chunk_1_it)

        self.assertEqual(self.detector.last_chunk, None)
        self.assertEqual(self.detector.actual_chunk, chunk_1_it)

        self.detector.update_chunks(chunk_2_it)

        self.assertEqual(self.detector.last_chunk, chunk_1_it)
        self.assertEqual(self.detector.actual_chunk, chunk_2_it)

    def test_increase_counter_should_increase_drift_counter_by_one(self):
        self.assertEqual(self.detector.drift_counter, 0)

        self.detector.increase_counter()

        self.assertEqual(self.detector.drift_counter, 1)

    def test_detect_drift_should_return_false_when_chunk_acc_is_higher_than_or_equal_to_threshold(  # NOQA
        self
    ):
        self.assertFalse(self.detector.detect(0.9))

    def test_detect_drift_should_return_true_when_chunk_acc_is_lower_than_threshold(  # NOQA
        self
    ):
        self.assertTrue(self.detector.detect(0.6))
