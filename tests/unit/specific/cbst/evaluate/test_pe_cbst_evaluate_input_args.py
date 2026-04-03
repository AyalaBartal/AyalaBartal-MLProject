import unittest
from types import SimpleNamespace


class CbstPeEvaluateInputArgs:
    """Input arguments for CatBoost evaluator."""

    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.input_csv = f"{input_dir}/data.csv"
        self.input_model = f"{input_dir}/model.joblib"


class TestCbstPeEvaluateInputArgs(unittest.TestCase):

    def test_init_sets_input_dir(self):
        input_dir = "/path/to/input"
        args = CbstPeEvaluateInputArgs(input_dir)

        self.assertEqual(args.input_dir, input_dir)

    def test_init_constructs_input_csv_path(self):
        input_dir = "/path/to/input"
        args = CbstPeEvaluateInputArgs(input_dir)

        self.assertEqual(args.input_csv, "/path/to/input/data.csv")

    def test_init_constructs_input_model_path(self):
        input_dir = "/path/to/input"
        args = CbstPeEvaluateInputArgs(input_dir)

        self.assertEqual(args.input_model, "/path/to/input/model.joblib")

    def test_input_csv_uses_standard_name(self):
        args = CbstPeEvaluateInputArgs("/data")

        self.assertTrue(args.input_csv.endswith("data.csv"))

    def test_input_model_uses_standard_name(self):
        args = CbstPeEvaluateInputArgs("/data")

        self.assertTrue(args.input_model.endswith("model.joblib"))

    def test_multiple_instances_with_different_dirs(self):
        args1 = CbstPeEvaluateInputArgs("/path1")
        args2 = CbstPeEvaluateInputArgs("/path2")

        self.assertEqual(args1.input_dir, "/path1")
        self.assertEqual(args2.input_dir, "/path2")
        self.assertNotEqual(args1.input_csv, args2.input_csv)

    def test_paths_maintain_directory_structure(self):
        input_dir = "/complex/nested/path"
        args = CbstPeEvaluateInputArgs(input_dir)

        self.assertEqual(args.input_csv, "/complex/nested/path/data.csv")
        self.assertEqual(args.input_model, "/complex/nested/path/model.joblib")


if __name__ == "__main__":
    unittest.main()
