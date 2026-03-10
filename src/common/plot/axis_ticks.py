"""
Represents tick positions and labels for a single axis. Example:
    positions=(0, 1)
    labels=("Negative", "Positive")
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class AxisTicks:
    positions: Tuple[int, ...]
    labels: Tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.positions) != len(self.labels):
            raise ValueError("positions and labels must have the same length")
