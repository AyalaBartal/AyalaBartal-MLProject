import unittest
from types import SimpleNamespace


class CbstPeEvaluateAlgoArgs:
    """Algorithm-specific evaluation arguments for CatBoost evaluator."""

    def __init__(self):
        self.column_label = 'Label'
        self.threshold = 0.5


class TestCbstPeEvaluateAlgoArgs(unittest.TestCase):

    def test_init_with_default_values(self):
        args = CbstPeEvaluateAlgoArgs()

        self.assertEqual(args.column_label, 'Label')
        self.assertEqual(args.threshold, 0.5)

    def test_can_modify_column_label(self):
        args = CbstPeEvaluateAlgoArgs()
        args.column_label = 'target'
        self.assertEqual(args.column_label, 'target')

    def test_can_modify_threshold(self):
        args = CbstPeEvaluateAlgoArgs()
        args.threshold = 0.7
        self.assertEqual(args.threshold, 0.7)

    def test_can_modify_all_attributes(self):
        args = CbstPeEvaluateAlgoArgs()
        args.column_label = 'custom_label'
        args.threshold = 0.6

        self.assertEqual(args.column_label, 'custom_label')
        self.assertEqual(args.threshold, 0.6)

    def test_threshold_can_be_set_to_various_values(self):
        args = CbstPeEvaluateAlgoArgs()
        
        args.threshold = 0.3
        self.assertEqual(args.threshold, 0.3)
        
        args.threshold = 0.9
        self.assertEqual(args.threshold, 0.9)

    def test_multiple_instances_independent(self):
        args1 = CbstPeEvaluateAlgoArgs()
        args2 = CbstPeEvaluateAlgoArgs()
        
        args1.column_label = 'custom_label'
        args1.threshold = 0.8
        
        self.assertEqual(args1.column_label, 'custom_label')
        self.assertEqual(args1.threshold, 0.8)
        self.assertEqual(args2.column_label, 'Label')
        self.assertEqual(args2.threshold, 0.5)


if __name__ == "__main__":
    unittest.main()
