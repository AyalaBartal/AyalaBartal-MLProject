
from dataclasses import dataclass


# Represents style settings for text annotations.
@dataclass(frozen=True)
class PlotTextStyle:
    horizontal_alignment: str = "center"
    vertical_alignment: str = "center"
    font_size: int = 11
    font_weight: str = "bold"
