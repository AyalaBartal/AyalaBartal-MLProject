import unittest
import numpy as np

from src.common.plot import TextAnnotation, ConfusionMatrixPlotSpec
from src.common.plot.confusion_matrix_spec_factory import ConfusionMatrixPlotSpecFactory


class TestConfusionMatrixPlotSpecFactory(unittest.TestCase):

    def setUp(self):
        self.factory = ConfusionMatrixPlotSpecFactory()

    def test_validate_confusion_matrix_data_none(self):
        with self.assertRaises(ValueError):
            self.factory.validate_confusion_matrix_data(None)

    def test_validate_confusion_matrix_data_wrong_type(self):
        with self.assertRaises(TypeError):
            self.factory.validate_confusion_matrix_data([[1, 2], [3, 4]])

    def test_validate_confusion_matrix_data_wrong_shape(self):
        cm = np.array([[1, 2, 3], [4, 5, 6]])

        with self.assertRaises(ValueError):
            self.factory.validate_confusion_matrix_data(cm)

    def test_validate_confusion_matrix_data_valid(self):
        cm = np.array([[1, 2], [3, 4]])

        # Should not raise
        self.factory.validate_confusion_matrix_data(cm)

    def test_get_annotations(self):
        cm = np.array([[10, 2], [3, 15]])

        annotations = self.factory.get_annotations(cm)

        self.assertEqual(len(annotations), 4)

        expected = [
            TextAnnotation(0, 0, "TN\n10"),
            TextAnnotation(1, 0, "FP\n2"),
            TextAnnotation(0, 1, "FN\n3"),
            TextAnnotation(1, 1, "TP\n15"),
        ]

        for a, e in zip(annotations, expected):
            self.assertEqual(a.x, e.x)
            self.assertEqual(a.y, e.y)
            self.assertEqual(a.text, e.text)

    def test_build_returns_plot_spec(self):
        cm = np.array([[5, 1], [2, 8]])

        spec = self.factory.build(cm)

        self.assertIsInstance(spec, ConfusionMatrixPlotSpec)
        self.assertTrue((spec.matrix == cm).all())
        self.assertEqual(spec.title, "Confusion Matrix — Decision Tree")

    def test_build_annotations_content(self):
        cm = np.array([[7, 3], [4, 9]])

        spec = self.factory.build(cm)

        texts = [a.text for a in spec.annotations]

        self.assertIn("TN\n7", texts)
        self.assertIn("FP\n3", texts)
        self.assertIn("FN\n4", texts)
        self.assertIn("TP\n9", texts)

    def test_build_axis_ticks(self):
        cm = np.array([[1, 2], [3, 4]])

        spec = self.factory.build(cm)

        self.assertEqual(spec.x_ticks.labels, ("Negative", "Positive"))
        self.assertEqual(spec.y_ticks.labels, ("Negative", "Positive"))

    def test_build_style(self):
        cm = np.array([[1, 2], [3, 4]])

        spec = self.factory.build(cm)

        self.assertEqual(spec.text_style.horizontal_alignment, "center")
        self.assertEqual(spec.text_style.vertical_alignment, "center")
        self.assertEqual(spec.text_style.font_size, 11)
        self.assertEqual(spec.text_style.font_weight, "bold")