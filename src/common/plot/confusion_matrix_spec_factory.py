from __future__ import annotations
import numpy as np
from src.common.plot import AxisTicks, TextAnnotation
from src.common.plot.plot_text_style import PlotTextStyle
from src.common.plot.confusion_matrix_plot_spec import ConfusionMatrixPlotSpec


"""
Factory responsible for generating ConfusionMatrixPlotSpec objects from a confusion matrix.
- Convert a numeric confusion matrix into a full plot specification
- including annotations, labels, ticks and visual settings.
"""
class ConfusionMatrixPlotSpecFactory:

    cm_labels = np.array([
        ["TN", "FP"],
        ["FN", "TP"],
    ])

    #  Validate the plot specification after object creation.
    def validate_confusion_matrix_data(self, cm):
        if cm is None:
            raise ValueError("cm must not be None")
        if not isinstance(cm, np.ndarray):
            raise TypeError("cm must be an instance of np.ndarray")
        if cm.shape != (2, 2):
            raise ValueError("cm must be a 2x2 confusion matrix")

    def build(self, cm_array: np.ndarray) -> ConfusionMatrixPlotSpec:
        cm_array = np.asarray(cm_array)
        cm_annotations = self.get_annotations(cm_array)
        return self.get_confusion_matrix_plot_spec(cm_annotations, cm_array)

    def get_confusion_matrix_plot_spec(self, cm_annotations, cm_array):
        return ConfusionMatrixPlotSpec(
            matrix=cm_array,
            annotations=cm_annotations,
            title="Confusion Matrix — Decision Tree",
            x_label="Predicted",
            y_label="Actual",
            x_ticks=AxisTicks(
                positions=(0, 1),
                labels=("Negative", "Positive"),
            ),
            y_ticks=AxisTicks(
                positions=(0, 1),
                labels=("Negative", "Positive"),
            ),
            text_style=PlotTextStyle(
                horizontal_alignment="center",
                vertical_alignment="center",
                font_size=11,
                font_weight="bold",
            ),
            cmap="Greens",
            fig_size=(4, 4),
            dpi=150,
            colorbar_fraction=0.046,
            colorbar_pad=0.04,
        )

    def get_annotations(self, cm_array):
        return tuple(
            TextAnnotation(
                x=j,
                y=i,
                text=f"{self.cm_labels[i, j]}\n{value}",
            )
            for (i, j), value in np.ndenumerate(cm_array)
        )
