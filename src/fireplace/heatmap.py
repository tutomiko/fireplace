import numpy as np
from scipy import ndimage


class ThermalIsland:
    """A single connected component of nonzero heat: its boolean mask
    (same shape as the parent heatmap grid) and its bounding box."""

    def __init__(self, mask: np.ndarray, bbox: tuple[int, int, int, int]):
        # mask: bool array, same (rows, cols) shape as the source heat grid,
        # True only where this island's pixels are.
        self._mask = mask
        # bbox: (row0, col0, row1, col1), row1/col1 exclusive, grid coordinates.
        self._bbox = bbox

    def get_mask(self) -> np.ndarray:
        return self._mask

    def get_bbox(self) -> tuple[int, int, int, int]:
        return self._bbox


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

    def get_islands(self) -> list[ThermalIsland]:
        """Splits the current heat mask into distinct thermal islands via
        connected-component labeling over all nonzero heat pixels. Each
        island is whatever remains connected at this frame -- callers that
        want a stricter or looser definition should threshold `get_mask()`
        upstream before constructing this FireplaceHeatmap."""
        binary = self._mask > 0
        if not np.any(binary):
            return []

        # 8-connectivity: diagonal neighbors count as connected, so islands
        # don't get needlessly split at pixel corners.
        structure = np.ones((3, 3), dtype=np.uint8)
        labeled, num_labels = ndimage.label(binary, structure=structure)

        islands = []
        for label_id in range(1, num_labels + 1):
            island_mask = labeled == label_id
            rows = np.any(island_mask, axis=1)
            cols = np.any(island_mask, axis=0)
            row0, row1 = np.where(rows)[0][[0, -1]]
            col0, col1 = np.where(cols)[0][[0, -1]]
            # Exclusive upper bound, consistent with typical bbox conventions.
            bbox = (int(row0), int(col0), int(row1) + 1, int(col1) + 1)
            islands.append(ThermalIsland(island_mask, bbox))

        return islands
