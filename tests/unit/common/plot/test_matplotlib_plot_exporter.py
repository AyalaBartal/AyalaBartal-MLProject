import os
import tempfile
import unittest
from matplotlib import pyplot as plt
from src.common.plot.matplotlib_plot_renderer import MatplotlibPlotRenderer
from src.common.plot.matplotlib_plot_exporter import MatplotlibPlotExporter
from tests.utils import ConfusionMatrixSpecProvider


class TestMatplotlibPlotExporter(unittest.TestCase):

    def setUp(self):
        self.renderer = MatplotlibPlotRenderer()
        self.exporter = MatplotlibPlotExporter(self.renderer)
        self.spec = ConfusionMatrixSpecProvider.get_spec()

    def tearDown(self):
        plt.close("all")

    def test_init_with_none_renderer_raises_value_error(self):
        with self.assertRaises(ValueError):
            MatplotlibPlotExporter(None)

    def test_init_with_wrong_renderer_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            MatplotlibPlotExporter(object())

    def test_render_and_save_with_none_spec_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_png = os.path.join(tmp_dir, "plot.png")

            with self.assertRaises(ValueError):
                self.exporter.render_and_save(None, out_png)

    def test_render_and_save_with_wrong_spec_type_raises_type_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_png = os.path.join(tmp_dir, "plot.png")

            with self.assertRaises(TypeError):
                self.exporter.render_and_save(object(), out_png)

    def test_render_and_save_writes_png_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_png = os.path.join(tmp_dir, "plot.png")

            self.exporter.render_and_save(self.spec, out_png)

            self.assertTrue(os.path.isfile(out_png))
            self.assertGreater(os.path.getsize(out_png), 0)

    def test_render_and_save_closes_figure_after_success(self):
        before = set(plt.get_fignums())

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_png = os.path.join(tmp_dir, "plot.png")
            self.exporter.render_and_save(self.spec, out_png)

        after = set(plt.get_fignums())
        self.assertEqual(before, after)
