from __future__ import annotations
import matplotlib.pyplot as plt

from src.common.plot.text_annotation import TextAnnotation
from src.common.plot.confusion_matrix_plot_spec import ConfusionMatrixPlotSpec


"""
Matplotlib rendering backend for confusion matrix plots.

This class is responsible only for translating a plot specification
(ConfusionMatrixPlotSpec) into actual matplotlib drawing operations.

Design goals:
- Isolate matplotlib dependency from the rest of the application
- Keep business logic outside of rendering
- Allow easy replacement of the rendering backend
- Allow easy mocking in unit tests
"""

"""
Concrete plotting backend that renders a ConfusionMatrixPlotSpec
using matplotlib.

Responsibilities:
- Create figure and axes
- Render the confusion matrix heatmap
- Render text annotations
- Configure axis titles and ticks
- Add colorbar
- Save the figure to disk

Non-responsibilities:
- Does not compute labels (TN/FP/FN/TP)
- Does not validate confusion matrix semantics
- Does not build plot specifications
"""


class MatplotlibPlotRenderer:

    """
    Render the given plot specification and save the resulting image.
    spec: Immutable plot description containing all data needed to render the confusion matrix.
    out_png: Output path for the generated PNG file.
    """
    def render_and_save(self, spec: ConfusionMatrixPlotSpec, out_png: str) -> None:
        fig, ax = plt.subplots(figsize=spec.fig_size)
        try:
            # Render confusion matrix heatmap
            im = ax.imshow(spec.matrix, cmap=spec.cmap)
            MatplotlibPlotRenderer.draw_all_spec_annotations(ax, spec)
            MatplotlibPlotRenderer.draw_all_titles_and_axis_labels(ax, spec)
            MatplotlibPlotRenderer.draw_all_axis_ticks(ax, spec)

            fig.colorbar(im, ax=ax, fraction=spec.colorbar_fraction, pad=spec.colorbar_pad)
            fig.tight_layout()
            fig.savefig(out_png, dpi=spec.dpi)

        finally:
            # Ensure figure is always closed to prevent memory leaks
            plt.close(fig)

    @staticmethod
    def draw_all_axis_ticks(ax, spec):
        ax.set_xticks(spec.x_ticks.positions)
        ax.set_yticks(spec.y_ticks.positions)
        ax.set_xticklabels(spec.x_ticks.labels)
        ax.set_yticklabels(spec.y_ticks.labels)

    @staticmethod
    def draw_all_titles_and_axis_labels(ax, spec):
        ax.set_title(spec.title)
        ax.set_xlabel(spec.x_label)
        ax.set_ylabel(spec.y_label)

    @staticmethod
    def draw_all_spec_annotations(ax, spec):
        for annotation in spec.annotations:
            MatplotlibPlotRenderer.draw_one_annotation(ax, annotation, spec)

    @staticmethod
    def draw_one_annotation(
        ax,
        annotation: TextAnnotation,
        spec: ConfusionMatrixPlotSpec,
    ) -> None:
        style = spec.text_style

        ax.text(
            annotation.x,
            annotation.y,
            annotation.text,
            ha=style.horizontal_alignment,
            va=style.vertical_alignment,
            fontsize=style.font_size,
            fontweight=style.font_weight,
        )
