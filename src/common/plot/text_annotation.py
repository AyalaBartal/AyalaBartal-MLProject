from __future__ import annotations
from dataclasses import dataclass

"""
Data class TextAnnotation is used in confusion matrix: confusion_matrix_plot_spec
TextAnnotation represents a single text annotation that should be drawn on the plot.
Example: x=0, y=0, text="TN\\n14"
Meaning:  Draw the text at matrix cell coordinates (x, y).
"""


@dataclass(frozen=True)
class TextAnnotation:
    x: int
    y: int
    text: str
