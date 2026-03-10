from __future__ import annotations
from __future__ import annotations

from abc import ABC, abstractmethod
from confusion_matrix_plot_spec import ConfusionMatrixPlotSpec

"""
Define PlotRender interface for confusion matrix graph plots.
It depends on the plot-spec data classes, but does not depend on matplotlib or any other concrete rendering library.
"""


class PlotRender(ABC):

    # Render the given plot using specification
    @abstractmethod
    def render(self, spec: ConfusionMatrixPlotSpec) -> None:
        raise NotImplementedError
