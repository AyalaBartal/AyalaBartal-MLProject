import unittest
from types import SimpleNamespace


class CbstPeEvaluateOutputArgs:
    """Output arguments for CatBoost evaluator."""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.out_json = f"{output_dir}/report.json"
        self.out_md = f"{output_dir}/report.md"
        self.out_png = f"{output_dir}/confusion_matrix.png"


class TestCbstPeEvaluateOutputArgs(unittest.TestCase):

    def test_init_sets_output_dir(self):
        output_dir = "/path/to/output"
        args = CbstPeEvaluateOutputArgs(output_dir)

        self.assertEqual(args.output_dir, output_dir)

    def test_init_constructs_out_json_path(self):
        output_dir = "/path/to/output"
        args = CbstPeEvaluateOutputArgs(output_dir)

        self.assertEqual(args.out_json, "/path/to/output/report.json")

    def test_init_constructs_out_md_path(self):
        output_dir = "/path/to/output"
        args = CbstPeEvaluateOutputArgs(output_dir)

        self.assertEqual(args.out_md, "/path/to/output/report.md")

    def test_init_constructs_out_png_path(self):
        output_dir = "/path/to/output"
        args = CbstPeEvaluateOutputArgs(output_dir)

        self.assertEqual(args.out_png, "/path/to/output/confusion_matrix.png")

    def test_out_json_uses_standard_name(self):
        args = CbstPeEvaluateOutputArgs("/output")

        self.assertTrue(args.out_json.endswith("report.json"))

    def test_out_md_uses_standard_name(self):
        args = CbstPeEvaluateOutputArgs("/output")

        self.assertTrue(args.out_md.endswith("report.md"))

    def test_out_png_uses_standard_name(self):
        args = CbstPeEvaluateOutputArgs("/output")

        self.assertTrue(args.out_png.endswith("confusion_matrix.png"))

    def test_multiple_instances_with_different_dirs(self):
        args1 = CbstPeEvaluateOutputArgs("/path1")
        args2 = CbstPeEvaluateOutputArgs("/path2")

        self.assertEqual(args1.output_dir, "/path1")
        self.assertEqual(args2.output_dir, "/path2")
        self.assertNotEqual(args1.out_json, args2.out_json)

    def test_paths_maintain_directory_structure(self):
        output_dir = "/complex/nested/output"
        args = CbstPeEvaluateOutputArgs(output_dir)

        self.assertEqual(args.out_json, "/complex/nested/output/report.json")
        self.assertEqual(args.out_md, "/complex/nested/output/report.md")
        self.assertEqual(args.out_png, "/complex/nested/output/confusion_matrix.png")


if __name__ == "__main__":
    unittest.main()
