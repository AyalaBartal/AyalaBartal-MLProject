import numpy as np

from src.common.plot import TextAnnotation, ConfusionMatrixPlotSpec, AxisTicks
from src.common.plot.plot_text_style import PlotTextStyle


class ConfusionMatrixSpecProvider:

    @staticmethod
    def get_spec():
        return ConfusionMatrixPlotSpec(
            matrix=np.array([[10, 2], [3, 15]]),
            annotations=(
                TextAnnotation(0, 0, "TN\n10"),
                TextAnnotation(1, 0, "FP\n2"),
                TextAnnotation(0, 1, "FN\n3"),
                TextAnnotation(1, 1, "TP\n15"),
            ),
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
