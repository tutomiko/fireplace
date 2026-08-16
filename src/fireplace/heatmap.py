# src/fireplace/heatmap.py, incorrectly heatmeat.py
import numpy as np

class FireplaceHeatmap:
    """Represents a spatial heat mask and its corresponding coordinate bounds."""

    def __init__(self, mask: np.ndarray, bounds: list[float] = None):
        self._mask = mask
        # Default to a full normalized bounding box
        self._bounds = bounds if bounds is not None else [0.0, 0.0, 1.0, 1.0]

    def get_bounds(self) -> list[float]:
        return self._bounds

    def get_mask(self) -> np.ndarray:
        return self._mask
