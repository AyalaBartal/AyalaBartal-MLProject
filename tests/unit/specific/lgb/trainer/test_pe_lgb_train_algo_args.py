import unittest
from types import SimpleNamespace

from src.specific.lgb.trainer.pe_lgb_train_algo_args import LgbPeTrainAlgoArgs


class TestLgbPeTrainAlgoArgs(unittest.TestCase):

    def test_init_with_default_values(self):
        args = LgbPeTrainAlgoArgs()

        self.assertEqual(args.label, "Label")
        self.assertEqual(args.n_estimators, 100)
        self.assertEqual(args.max_depth, 6)
        self.assertEqual(args.learning_rate, 0.1)
        self.assertEqual(args.num_leaves, 31)
        self.assertEqual(args.random_state, 42)
        self.assertEqual(args.n_splits, 10)

    def test_can_modify_label(self):
        args = LgbPeTrainAlgoArgs()
        args.label = "target"
        self.assertEqual(args.label, "target")

    def test_can_modify_hyperparameters(self):
        args = LgbPeTrainAlgoArgs()
        args.n_estimators = 200
        args.max_depth = 8
        args.learning_rate = 0.05
        
        self.assertEqual(args.n_estimators, 200)
        self.assertEqual(args.max_depth, 8)
        self.assertEqual(args.learning_rate, 0.05)

    def test_can_modify_num_leaves(self):
        args = LgbPeTrainAlgoArgs()
        args.num_leaves = 63
        self.assertEqual(args.num_leaves, 63)

    def test_can_modify_n_splits(self):
        args = LgbPeTrainAlgoArgs()
        args.n_splits = 5
        self.assertEqual(args.n_splits, 5)

    def test_can_modify_random_state(self):
        args = LgbPeTrainAlgoArgs()
        args.random_state = 123
        self.assertEqual(args.random_state, 123)

    def test_multiple_instances_independent(self):
        args1 = LgbPeTrainAlgoArgs()
        args2 = LgbPeTrainAlgoArgs()
        
        args1.label = "custom_label"
        
        self.assertEqual(args1.label, "custom_label")
        self.assertEqual(args2.label, "Label")


if __name__ == "__main__":
    unittest.main()
