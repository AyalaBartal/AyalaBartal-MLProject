from __future__ import annotations
from __future__ import annotations
from abc import ABC, abstractmethod
from confusion_matrix_plot_spec import ConfusionMatrixPlotSpec

# Define PlotExporter interface for confusion matrix graph plots.
class PlotExporter(ABC):
    """
    Render the given plot specification and save it to the output path. Parameters
    spec: Immutable description of what should be drawn.
    out_png: Output file path for the generated image.
    """

    # Render the given plot using specification and save into file out_png
    @abstractmethod
    def render_and_save(self, spec: ConfusionMatrixPlotSpec, out_png: str):
        raise NotImplementedError

