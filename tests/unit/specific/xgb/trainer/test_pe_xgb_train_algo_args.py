import unittest
from types import SimpleNamespace

from src.specific.xgb.trainer.pe_xgb_train_algo_args import XgbPeTrainAlgoArgs


class TestXgbPeTrainAlgoArgs(unittest.TestCase):

    def test_init_with_default_values(self):
        args = XgbPeTrainAlgoArgs()

        self.assertEqual(args.label, "Label")
        self.assertEqual(args.n_estimators, 100)
        self.assertEqual(args.max_depth, 6)
        self.assertEqual(args.learning_rate, 0.1)
        self.assertEqual(args.subsample, 0.8)
        self.assertEqual(args.colsample_bytree, 0.8)
        self.assertEqual(args.scale_pos_weight, 1.0)
        self.assertEqual(args.random_state, 42)
        self.assertEqual(args.n_splits, 10)

    def test_can_modify_label(self):
        args = XgbPeTrainAlgoArgs()
        args.label = "target"
        self.assertEqual(args.label, "target")

    def test_can_modify_hyperparameters(self):
        args = XgbPeTrainAlgoArgs()
        args.n_estimators = 200
        args.max_depth = 8
        args.learning_rate = 0.05
        
        self.assertEqual(args.n_estimators, 200)
        self.assertEqual(args.max_depth, 8)
        self.assertEqual(args.learning_rate, 0.05)

    def test_can_modify_subsample(self):
        args = XgbPeTrainAlgoArgs()
        args.subsample = 0.9
        self.assertEqual(args.subsample, 0.9)

    def test_can_modify_colsample_bytree(self):
        args = XgbPeTrainAlgoArgs()
        args.colsample_bytree = 0.7
        self.assertEqual(args.colsample_bytree, 0.7)

    def test_can_modify_scale_pos_weight(self):
        args = XgbPeTrainAlgoArgs()
        args.scale_pos_weight = 2.0
        self.assertEqual(args.scale_pos_weight, 2.0)

    def test_can_modify_n_splits(self):
        args = XgbPeTrainAlgoArgs()
        args.n_splits = 5
        self.assertEqual(args.n_splits, 5)

    def test_can_modify_random_state(self):
        args = XgbPeTrainAlgoArgs()
        args.random_state = 123
        self.assertEqual(args.random_state, 123)

    def test_multiple_instances_independent(self):
        args1 = XgbPeTrainAlgoArgs()
        args2 = XgbPeTrainAlgoArgs()
        
        args1.label = "custom_label"
        
        self.assertEqual(args1.label, "custom_label")
        self.assertEqual(args2.label, "Label")


if __name__ == "__main__":
    unittest.main()

