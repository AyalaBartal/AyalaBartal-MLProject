from __future__ import annotations

import matplotlib.pyplot as plt

from .confusion_matrix_plot_spec import ConfusionMatrixPlotSpec
from .matplotlib_plot_renderer import MatplotlibPlotRenderer


# Export a plot ConfusionMatrixPlotSpec to a PNG file using matplotlib.
class MatplotlibPlotExporter:

    def __init__(self, renderer: MatplotlibPlotRenderer):
        MatplotlibPlotExporter.validate_renderer(renderer)
        self.renderer = renderer

    # Export by render and then save and ensure figure is closed even if saving fails
    def render_and_save(self, spec: ConfusionMatrixPlotSpec, out_png: str):
        self.validate_spec(spec)

        fig = self.renderer.render(spec)
        try:
            fig.savefig(out_png, dpi=spec.dpi)
        finally:
            plt.close(fig)

    @staticmethod
    def validate_renderer(renderer):
        if renderer is None:
            raise ValueError("renderer must not be None")
        if not isinstance(renderer, MatplotlibPlotRenderer):
            raise TypeError("renderer must be an instance of MatplotlibPlotRenderer")

    @staticmethod
    def validate_spec(spec):
        if spec is None:
            raise ValueError("spec must not be None")
        if not isinstance(spec, ConfusionMatrixPlotSpec):
            raise TypeError("spec must be ConfusionMatrixPlotSpec")
