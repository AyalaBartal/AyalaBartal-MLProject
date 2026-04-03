import unittest
from types import SimpleNamespace


class CbstPeTrainAlgoArgs:
    """Algorithm-specific training arguments for CatBoost trainer."""

    def __init__(self):
        self.label = "Label"
        self.n_estimators = 100
        self.max_depth = 6
        self.learning_rate = 0.1
        self.num_leaves = 31
        self.random_state = 42
        self.n_splits = 10


class TestCbstPeTrainAlgoArgs(unittest.TestCase):

    def test_init_with_default_values(self):
        args = CbstPeTrainAlgoArgs()

        self.assertEqual(args.label, "Label")
        self.assertEqual(args.n_estimators, 100)
        self.assertEqual(args.max_depth, 6)
        self.assertEqual(args.learning_rate, 0.1)
        self.assertEqual(args.num_leaves, 31)
        self.assertEqual(args.random_state, 42)
        self.assertEqual(args.n_splits, 10)

    def test_can_modify_label(self):
        args = CbstPeTrainAlgoArgs()
        args.label = "target"
        self.assertEqual(args.label, "target")

    def test_can_modify_hyperparameters(self):
        args = CbstPeTrainAlgoArgs()
        args.n_estimators = 200
        args.max_depth = 8
        args.learning_rate = 0.05
        
        self.assertEqual(args.n_estimators, 200)
        self.assertEqual(args.max_depth, 8)
        self.assertEqual(args.learning_rate, 0.05)

    def test_can_modify_num_leaves(self):
        args = CbstPeTrainAlgoArgs()
        args.num_leaves = 63
        self.assertEqual(args.num_leaves, 63)

    def test_can_modify_n_splits(self):
        args = CbstPeTrainAlgoArgs()
        args.n_splits = 5
        self.assertEqual(args.n_splits, 5)

    def test_can_modify_random_state(self):
        args = CbstPeTrainAlgoArgs()
        args.random_state = 123
        self.assertEqual(args.random_state, 123)

    def test_multiple_instances_independent(self):
        args1 = CbstPeTrainAlgoArgs()
        args2 = CbstPeTrainAlgoArgs()
        
        args1.label = "custom_label"
        
        self.assertEqual(args1.label, "custom_label")
        self.assertEqual(args2.label, "Label")


if __name__ == "__main__":
    unittest.main()
