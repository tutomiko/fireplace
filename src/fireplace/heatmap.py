import numpy as np
from scipy.ndimage import gaussian_filter, label, maximum_filter


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

    def get_eyes(self, sigma: float = None) -> list[tuple[float, float]]:
        """
        Detects heat peaks simultaneously across the heatmap using Gaussian smoothing
        and local maxima filtering. Returns normalized (0.0-1.0) coordinates for all
        peaks exceeding 50% of their island's maximum heat.
        """
        if self._mask is None or self._mask.size == 0:
            return []

        mask = np.atleast_2d(self._mask)
        if np.max(mask) <= 0:
            return []

        bounds = self._bounds if self._bounds is not None else [0.0, 0.0, 1.0, 1.0]
        x_min, y_min, x_max, y_max = bounds
        h, w = mask.shape

        # 1. Smooth scattered stipple dots into continuous density hills
        if sigma is None:
            sigma = max(3.0, min(h, w) * 0.025)

        smoothed = gaussian_filter(mask.astype(float), sigma=sigma)

        global_max = float(np.max(smoothed))
        if global_max <= 1e-6:
            return []

        structure = np.ones((3, 3), dtype=int)

        # 2. Identify connected heat islands (ignoring low background noise)
        island_mask = smoothed > (0.01 * global_max)
        labeled_islands, num_islands = label(island_mask, structure=structure)
        if num_islands == 0:
            return []

        # Store max heat value per island
        island_peaks = {}
        for island_id in range(1, num_islands + 1):
            island_peaks[island_id] = float(np.max(smoothed[labeled_islands == island_id]))

        # 3. Detect all local peaks simultaneously using maximum_filter
        window_size = max(7, int(sigma * 2) | 1)
        local_max_bool = (maximum_filter(smoothed, size=window_size) == smoothed) & island_mask

        # Label connected peak components (prevents flat peak plateaus from splintering)
        labeled_peaks, num_peaks = label(local_max_bool, structure=structure)
        if num_peaks == 0:
            return []

        eyes = []

        # 4. Filter and normalize coordinates for valid eyes (>= 50% of island peak)
        for peak_id in range(1, num_peaks + 1):
            peak_coords = np.argwhere(labeled_peaks == peak_id)
            if len(peak_coords) == 0:
                continue

            r_sample, c_sample = peak_coords[0]
            island_id = labeled_islands[r_sample, c_sample]
            if island_id == 0:
                continue

            peak_val = float(smoothed[r_sample, c_sample])
            island_peak_val = island_peaks.get(island_id, 0.0)

            if peak_val >= 0.5 * island_peak_val:
                avg_r = float(np.mean(peak_coords[:, 0]))
                avg_c = float(np.mean(peak_coords[:, 1]))

                norm_x = x_min + (avg_c / (w - 1)) * (x_max - x_min) if w > 1 else x_min + 0.5 * (x_max - x_min)
                norm_y = y_min + (avg_r / (h - 1)) * (y_max - y_min) if h > 1 else y_min + 0.5 * (y_max - y_min)

                norm_x = float(np.clip(norm_x, x_min, x_max))
                norm_y = float(np.clip(norm_y, y_min, y_max))

                eyes.append((norm_x, norm_y))

        return eyes
