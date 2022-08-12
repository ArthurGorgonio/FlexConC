from unittest import TestCase

from mock import Mock, patch

from src.ssl.flexcon import BaseFlexConC


class SelfTrainingClassifierMock(Mock):
    ...


class GenerateMemory():
    def pred_1_it(self):
        return {
            1: {
                'confidence': 0.3,
                'classes': 0
            },
            2: {
                'confidence': 0.2,
                'classes': 0
            },
            3: {
                'confidence': 0.97,
                'classes': 0
            },
            4: {
                'confidence': 0.97,
                'classes': 1
            },
            5: {
                'confidence': 0.98,
                'classes': 1
            }
        }

    def pred_x_it(self):
        return {
            1: {
                'confidence': 0.3,
                'classes': 0
            },
            2: {
                'confidence': 0.92,
                'classes': 1
            },
            3: {
                'confidence': 0.96,
                'classes': 1
            },
            4: {
                'confidence': 0.7,
                'classes': 1
            },
            5: {
                'confidence': 0.99,
                'classes': 1
            }
        }


class TestFlexCon(TestCase):
    @patch("src.ssl.flexcon.clone")
    @patch("src.ssl.flexcon.SelfTrainingClassifier")
    def setUp(self, super_class, model_clone):
        super_class.return_value = SelfTrainingClassifierMock()
        model_clone.return_value = ""

        self.flexcon = BaseFlexConC("")

    def test_should_return_updated_cl_memory_by_weights_when_weights_are_passed(self):  # NOQA
        instances = list(range(10))
        labels = [0, 0, 0, 1, 1, 1, 1, 1, 0, 1]
        weights = [0.3, 0.2, 0.5, 0.6, 0.7, 0.1, 0.8, 0.4, 0.9, 0.57]
        expected_output_weights = [
            [0.3, 0],
            [0.2, 0],
            [0.5, 0],
            [0, 0.6],
            [0, 0.7],
            [0, 0.1],
            [0, 0.8],
            [0, 0.4],
            [0.9, 0],
            [0, 0.57],
        ]
        self.flexcon.cl_memory = [[0] * 2 for _ in range(len(instances))]
        self.flexcon.update_memory(instances, labels, weights)
        self.assertListEqual(self.flexcon.cl_memory, expected_output_weights)

    def test_should_return_updated_cl_memory_by_one_when_no_weights_are_passed(self):  # NOQA
        instances = list(range(10))
        labels = [0, 0, 0, 1, 1, 1, 1, 1, 0, 1]
        labels2 = [1, 1, 0, 1, 0, 1, 0, 0, 1, 1]
        output_without_weights = [
            [1, 1],
            [1, 1],
            [2, 0],
            [0, 2],
            [1, 1],
            [0, 2],
            [1, 1],
            [1, 1],
            [1, 1],
            [0, 2],
        ]
        self.flexcon.cl_memory = [[0] * 2 for _ in range(len(instances))]
        self.flexcon.update_memory(instances, labels)
        self.flexcon.update_memory(instances, labels2)
        self.assertListEqual(self.flexcon.cl_memory, output_without_weights)

    def test_rule1_should_return_single_instance_when_thr_90(self):
        self.flexcon.threshold = 0.9
        preds = GenerateMemory()
        self.flexcon.pred_x_it = preds.pred_x_it()
        self.flexcon.pred_1_it = preds.pred_1_it()
        # labels from dict (class1 == class2)
        expected_rule1 = ([5], [1])

        self.assertTupleEqual(self.flexcon.rule_1(), expected_rule1)

    def test_rule2_should_return_pair_instance_when_thr_90(self):
        self.flexcon.threshold = 0.9
        preds = GenerateMemory()
        self.flexcon.pred_x_it = preds.pred_x_it()
        self.flexcon.pred_1_it = preds.pred_1_it()
        # labels from dict (class1 == class2)
        expected_rule2 = ([4, 5], [1, 1])

        self.assertTupleEqual(self.flexcon.rule_2(), expected_rule2)

    @patch("src.ssl.flexcon.BaseFlexConC.remember")
    def test_rule3_should_return_single_instance_when_thr_90(self, remaind):
        remaind.return_value = [0]
        self.flexcon.threshold = 0.9
        preds = GenerateMemory()
        self.flexcon.pred_x_it = preds.pred_x_it()
        self.flexcon.pred_1_it = preds.pred_1_it()
        # labels from mock (class1 != class2)
        expected_rule3 = ([3], [0])

        self.assertTupleEqual(self.flexcon.rule_3(), expected_rule3)

    @patch("src.ssl.flexcon.BaseFlexConC.remember")
    def test_rule4_should_return_pair_instance_when_thr_90(self, remaind):
        remaind.return_value = [0]
        self.flexcon.threshold = 0.9
        preds = GenerateMemory()
        self.flexcon.pred_x_it = preds.pred_x_it()
        self.flexcon.pred_1_it = preds.pred_1_it()
        # labels from mock (class1 != class2)
        expected_rule4 = ([2, 3], [0])

        self.assertTupleEqual(self.flexcon.rule_4(), expected_rule4)

    @patch("src.ssl.flexcon.BaseFlexConC.remember")
    def test_select_instances_should_return_empty_numpy_array_when_no_instance_is_available_for_insertion_rules(self, remaind):  # NOQA
        self.flexcon.threshold = 1.0
        preds = GenerateMemory()
        self.flexcon.pred_x_it = preds.pred_x_it()
        self.flexcon.pred_1_it = preds.pred_1_it()
        expected_output = ""

        _, pred = self.flexcon.select_instances_by_rules()

        self.assertEqual(pred, expected_output)
        self.assertEqual(remaind.call_count, 2)

    def test_new_threshold_should_return_higher_threshold_when_local_acc_is_lower_than_init_acc(self):  # NOQA
        self.flexcon.new_threshold(0.4, 0.9)

        self.assertEqual(self.flexcon.threshold, 1.0)

    def test_new_threshold_should_return_lower_threshold_when_local_acc_is_higher_than_init_acc(self):  # NOQA
        self.flexcon.new_threshold(1.0, 0.9)

        self.assertEqual(self.flexcon.threshold, 0.8999999999999999)

    def test_new_threshold_should_return_unchanged_threshold_when_local_acc_in_acceptable_variace_from_init_acc(self):  # NOQA
        self.flexcon.new_threshold(0.9, 0.9)

        self.assertEqual(self.flexcon.threshold, 0.95)

        self.flexcon.new_threshold(0.91, 0.9)

        self.assertEqual(self.flexcon.threshold, 0.95)

        self.flexcon.new_threshold(0.89, 0.9)

        self.assertEqual(self.flexcon.threshold, 0.95)

    def test_remember_should_return_best_labelled_classes_for_each_instance(self):  # NOQA
        expected_output = [0, 0, 2, 1, 2, 0, 2, 1, 1, 2]
        self.flexcon.cl_memory = [
            [5, 0, 3],
            [5, 5, 5],
            [1, 2, 3],
            [2, 4, 3],
            [0, 1, 2],
            [3, 2, 0],
            [0, 0, 3],
            [3, 4, 2],
            [0, 5, 3],
            [4, 2, 5],
        ]
        instances = list(range(10))
        labels = self.flexcon.remember(instances)

        self.assertListEqual(labels, expected_output)

    def test_storage_predict_should_return_valid_struct(self,):
        expected_output = {
            1: {
                'confidence': 0.952,
                'classes': 0
            },
            2: {
                'confidence': 0.45,
                'classes': 1
            },
        }

        instances = [1, 2]
        confidences = [0.952, 0.45]
        classes = [0, 1]

        self.assertDictEqual(
            self.flexcon.storage_predict(instances, confidences, classes),
            expected_output
        )
