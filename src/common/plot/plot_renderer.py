from __future__ import annotations
from __future__ import annotations

from abc import ABC, abstractmethod
from confusion_matrix_plot_spec import ConfusionMatrixPlotSpec

"""
Define PlotRender interface for confusion matrix graph plots.
It depends on the plot-spec data classes, but does not depend on matplotlib or any other concrete rendering library.
"""


class PlotRender(ABC):
    """
    Render the given plot specification and save it to the output path. Parameters
    spec: Immutable description of what should be drawn.
    out_png: Output file path for the generated image.
    """
    @abstractmethod
    def render_and_save(self, spec: ConfusionMatrixPlotSpec, out_png: str) -> None:
        raise NotImplementedError
