import unittest
import matplotlib
import matplotlib.pyplot as plt

from src.common.plot.matplotlib_plot_renderer import MatplotlibPlotRenderer
from src.common.plot.text_annotation import TextAnnotation
from tests.utils.confusion_matrix_spec_provider import ConfusionMatrixSpecProvider

matplotlib.use("Agg")


class TestMatplotlibPlotRenderer(unittest.TestCase):

    def setUp(self):
        self.renderer = MatplotlibPlotRenderer()
        self.spec = ConfusionMatrixSpecProvider.get_spec()


    def tearDown(self):
        plt.close("all")

    def test_render_returns_figure(self):
        fig = self.renderer.render(self.spec)

        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 2)  # main axis + colorbar

    def test_render_sets_titles_and_labels(self):
        fig = self.renderer.render(self.spec)
        ax = fig.axes[0]

        self.assertEqual(ax.get_title(), "Confusion Matrix — Decision Tree")
        self.assertEqual(ax.get_xlabel(), "Predicted")
        self.assertEqual(ax.get_ylabel(), "Actual")

    def test_render_sets_axis_ticks(self):
        fig = self.renderer.render(self.spec)
        ax = fig.axes[0]

        self.assertEqual(list(ax.get_xticks()), [0, 1])
        self.assertEqual(list(ax.get_yticks()), [0, 1])

    def test_render_sets_tick_labels(self):
        fig = self.renderer.render(self.spec)
        ax = fig.axes[0]

        self.assertEqual(
            [tick.get_text() for tick in ax.get_xticklabels()],
            ["Negative", "Positive"],
        )

        self.assertEqual(
            [tick.get_text() for tick in ax.get_yticklabels()],
            ["Negative", "Positive"],
        )

    def test_render_draws_all_annotations(self):
        fig = self.renderer.render(self.spec)
        ax = fig.axes[0]

        texts = [t.get_text() for t in ax.texts]

        self.assertEqual(len(texts), 4)
        self.assertIn("TN\n10", texts)
        self.assertIn("FP\n2", texts)
        self.assertIn("FN\n3", texts)
        self.assertIn("TP\n15", texts)

    def test_draw_one_annotation(self):
        fig, ax = plt.subplots()

        annotation = TextAnnotation(1, 0, "FP\n2")

        self.renderer.draw_one_annotation(ax, annotation, self.spec)

        self.assertEqual(len(ax.texts), 1)
        self.assertEqual(ax.texts[0].get_text(), "FP\n2")
        self.assertEqual(ax.texts[0].get_position(), (1, 0))

    def test_draw_all_spec_annotations(self):
        fig, ax = plt.subplots()

        self.renderer.draw_all_spec_annotations(ax, self.spec)

        texts = [t.get_text() for t in ax.texts]

        self.assertEqual(len(texts), 4)
        self.assertEqual(texts, ["TN\n10", "FP\n2", "FN\n3", "TP\n15"])

    def test_draw_all_titles_and_axis_labels(self):
        fig, ax = plt.subplots()

        self.renderer.draw_all_titles_and_axis_labels(ax, self.spec)

        self.assertEqual(ax.get_title(), "Confusion Matrix — Decision Tree")
        self.assertEqual(ax.get_xlabel(), "Predicted")
        self.assertEqual(ax.get_ylabel(), "Actual")

    def test_draw_all_axis_ticks(self):
        fig, ax = plt.subplots()

        self.renderer.draw_all_axis_ticks(ax, self.spec)

        self.assertEqual(list(ax.get_xticks()), [0, 1])
        self.assertEqual(list(ax.get_yticks()), [0, 1])

        self.assertEqual(
            [t.get_text() for t in ax.get_xticklabels()],
            ["Negative", "Positive"],
        )

        self.assertEqual(
            [t.get_text() for t in ax.get_yticklabels()],
            ["Negative", "Positive"],
        )
