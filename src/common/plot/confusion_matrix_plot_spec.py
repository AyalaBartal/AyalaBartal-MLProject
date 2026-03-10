from dataclasses import dataclass, field
from typing import Tuple
import numpy as np

from src.common.plot import AxisTicks, TextAnnotation
from src.common.plot.plot_text_style import PlotTextStyle

"""
    Immutable description of the confusion-matrix plot.
    This class answers: "What should be drawn?"
    It does NOT answer: "How should matplotlib, or any other impl, draw it?"
    
    A confusion matrix is result of testing AI model on input-expected-output and has 4 values:
    TN(True Negative): The model correctly predicts the negative
    FP(False Positive)
    FN(False Negative)
    TP(True Positive):  The model correctly predicts the positive

    Fields summary:
        matrix: The numeric confusion matrix data, expected to be 2x2.
        annotations: All text labels that should appear on top of the matrix cells.
        title: Plot title.
        x_label: Label for the X axis.
        y_label: Label for the Y axis.
        x_ticks: Tick positions and labels for the X axis.
        y_ticks: Tick positions and labels for the Y axis.
        text_style:  Style to apply to annotation text.
        cmap: Name of the color map to use for the heatmap.
        fig_size: Requested figure size in inches.
        dpi: Resolution for saved output image.
        colorbar_fraction: Fraction argument to pass to the rendering backend for colorbar size.
        colorbar_pad: Pad argument to pass to the rendering backend for colorbar spacing.
    """


@dataclass(frozen=True)
class ConfusionMatrixPlotSpec:
    matrix: np.ndarray
    annotations: Tuple[TextAnnotation, ...] = field(default_factory=tuple)

    title: str = "Confusion Matrix — Decision Tree"
    x_label: str = "Predicted"
    y_label: str = "Actual"

    x_ticks: AxisTicks = field(
        default_factory=lambda: AxisTicks(
            positions=(0, 1),
            labels=("Negative", "Positive"),
        )
    )
    y_ticks: AxisTicks = field(
        default_factory=lambda: AxisTicks(
            positions=(0, 1),
            labels=("Negative", "Positive"),
        )
    )

    text_style: PlotTextStyle = field(default_factory=PlotTextStyle)

    cmap: str = "Greens"
    fig_size: Tuple[int, int] = (4, 4)
    dpi: int = 150

    colorbar_fraction: float = 0.046
    colorbar_pad: float = 0.04

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix)

        # Ensure stored matrix is always a numpy array, even if caller passed a list.
        object.__setattr__(self, "matrix", matrix)

        if self.fig_size[0] <= 0 or self.fig_size[1] <= 0:
            raise ValueError("fig_size values must be positive")

        if self.dpi <= 0:
            raise ValueError("dpi must be positive")

        if self.colorbar_fraction <= 0:
            raise ValueError("colorbar_fraction must be positive")

        if self.colorbar_pad < 0:
            raise ValueError("colorbar_pad must be >= 0")

