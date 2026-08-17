import time
import cv2
import numpy as np
from typing import Callable, Optional

from .heatmap import FireplaceHeatmap


class Fireplace:
    """
    Induces, enriches, and dissipates spatial heatmaps based on embedding similarities.
    Calls `on_heatmap_changed` iteratively if a rendering clock is set.
    """

    def __init__(self):
        # Callbacks
        self.on_heatmap_changed: Optional[Callable[[FireplaceHeatmap], None]] = None
        self._centroids_provider: Optional[Callable[[], np.ndarray]] = None
        self._patch_embeddings_provider: Optional[Callable[[], np.ndarray]] = None
        
        # Parameters
        self._decay: float = 0.01
        self._decay_threshold: float = 0.2
        self._rendering_clock: float = 0.0
        
        # State
        self._embedding_space: Optional[np.ndarray] = None
        self._current_heat: Optional[np.ndarray] = None
        self._bounds: list[float] = [0.0, 0.0, 1.0, 1.0]

    def set_class_precomputed_centroids_provider(self, callback: Callable[[], np.ndarray]) -> None:
        self._centroids_provider = callback

    def set_class_nn_patch_embeddings_provider(self, callback: Callable[[], np.ndarray]) -> None:
        self._patch_embeddings_provider = callback

    def set_decay(self, decay: float) -> None:
        self._decay = decay

    def set_decay_threshold(self, threshold: float) -> None:
        self._decay_threshold = threshold

    def set_heatmap_rendering_clock(self, seconds: float) -> None:
        self._rendering_clock = seconds

    def set_embedding_space(self, embeddings_grid: np.ndarray) -> None:
        self._embedding_space = embeddings_grid

    def get_heatmap(self) -> FireplaceHeatmap:
        if self._current_heat is None:
            mask = np.zeros((1, 1), dtype=np.float32)
            return FireplaceHeatmap(mask, self._bounds)
            
        grid = self._current_heat
        nonzero = grid[grid > 0]

        if nonzero.size == 0:
            normalized = grid.copy()
        else:
            min_val = float(nonzero.min())
            max_val = float(nonzero.max())
            if max_val > min_val:
                normalized = np.where(grid > 0, (grid - min_val) / (max_val - min_val), 0.0)
            else:
                normalized = np.where(grid > 0, 1.0, 0.0)
                
        return FireplaceHeatmap(normalized.astype(np.float32), self._bounds)

    def _notify_change(self) -> None:
        if self.on_heatmap_changed is not None:
            self.on_heatmap_changed(self.get_heatmap())
        if self._rendering_clock > 0:
            time.sleep(self._rendering_clock)

    def form_centroid_heatmap(self) -> None:
        """Calculates multi-hotspot persistence across nearest neighbor centroids."""
        if self._embedding_space is None or self._centroids_provider is None:
            return

        centroids = self._centroids_provider()
        if centroids is None or len(centroids) == 0:
            return

        grid_h, grid_w, emb_dim = self._embedding_space.shape
        flat_embeddings = self._embedding_space.reshape(-1, emb_dim)
        
        accumulated_heat = np.zeros((grid_h, grid_w), dtype=np.float32)
        emb_norms = np.linalg.norm(flat_embeddings, axis=1, keepdims=True)
        emb_norms[emb_norms == 0] = 1e-6
        
        for idx, proto_vec in enumerate(centroids):
            round_idx = idx + 1
            proto = np.atleast_2d(proto_vec)
            proto_norm = np.linalg.norm(proto, axis=1, keepdims=True).T
            
            denom = np.dot(emb_norms, proto_norm)
            denom[denom == 0] = 1e-6
            
            sims = np.dot(flat_embeddings, proto.T) / denom
            heatmap_grid = sims.reshape(grid_h, grid_w)
            heatmap_grid = np.clip(heatmap_grid, 0.0, None)
            
            # Multi-hotspot persistence calculation
            accumulated_heat = ((accumulated_heat * idx) + heatmap_grid) / round_idx
            
            self._current_heat = accumulated_heat
            self._notify_change()

    def enrich_heat(self) -> None:
        """
        Narrows down heatmaps by filtering current heat through patch-level similarity.
        Non-matching regions drop exponentially based on exact cosine similarity.
        """
        if self._embedding_space is None or self._current_heat is None or self._patch_embeddings_provider is None:
            return

        proto_patches = self._patch_embeddings_provider()
        if proto_patches is None or len(proto_patches) == 0:
            return

        grid_h, grid_w, emb_dim = self._embedding_space.shape
        flat_grid = self._embedding_space.reshape(-1, emb_dim)
        grid_norms = np.linalg.norm(flat_grid, axis=1, keepdims=True)
        grid_norms = np.where(grid_norms == 0, 1e-6, grid_norms)
        
        proto_arr = np.array(proto_patches, dtype=np.float32)
        proto_norms = np.linalg.norm(proto_arr, axis=1, keepdims=True)
        proto_norms = np.where(proto_norms == 0, 1e-6, proto_norms)

        dot_product = np.dot(flat_grid, proto_arr.T)
        denom = np.dot(grid_norms, proto_norms.T)
        sim_matrix = dot_product / denom

        max_sim_grid = np.max(sim_matrix, axis=1).reshape(grid_h, grid_w)
        max_sim_grid = np.clip(max_sim_grid, 0.0, 1.0)
        
        # Sharp similarity response curve drops similarity below threshold exponentially
        filtered_similarity = np.where(
            max_sim_grid >= self._decay_threshold,
            ((max_sim_grid - self._decay_threshold) / (1.0 - self._decay_threshold)) ** 2,
            0.0
        )

        initial_heat = self._current_heat.copy()
        num_steps = 5

        # Stream progressive heat dispersal step-by-step
        for step in range(1, num_steps + 1):
            alpha = step / float(num_steps)

            # Smooth transition from original wide heat map to pinpoint patch-matched heatmap
            step_grid = initial_heat * ((1.0 - alpha) + (alpha * filtered_similarity))

            self._current_heat = step_grid
            self._notify_change()

    def dissipate_heat(self, phase: str = "unspecified") -> None:
        """
        Progressively elevates the floor cutoff so cooler heat regions dissolve away
        across multiple simulated steps.

        `phase` labels which point in the induce() lifecycle this call
        represents (e.g. "pre_enrich" / "post_enrich") -- dissipate_heat is
        called twice per induce() on different inputs (raw centroid heat vs
        enrich_heat's filtered output).
        """
        if self._current_heat is None:
            return

        base_grid = self._current_heat.copy()
        max_heat_before = float(np.max(base_grid)) if base_grid.size > 0 else 0.0

        if max_heat_before <= 0:
            self._notify_change()
            return

        target_cutoff = self._decay_threshold * max_heat_before
        dissipate_steps = 20

        # Stream smooth visible dissipation steps over time
        for step in range(1, dissipate_steps + 1):
            progress = step / dissipate_steps
            current_cutoff = progress * target_cutoff
            step_grid = base_grid.copy()

            # Apply exponential contract on lower intensities so cooler regions fade faster
            step_grid = np.where(step_grid < current_cutoff, 0.0, step_grid)

            # Smoothly decay intensity values of remaining hotspots
            step_grid *= (1.0 - (self._decay * progress))

            self._current_heat = step_grid
            self._notify_change()

    def smooth_heatmap(
        self,
        step_percent: float = 0.01,
        max_iterations: int = 150,
        max_cell_heat: float = 1.0,
    ) -> None:
        """
        Optional refinement pass: iteratively redistributes heat from hot
        cells toward their cooler, inward-facing neighbors within each
        connected island, smoothing out jagged single-cell spikes.

        This is not part of the induce() lifecycle -- callers opt in by
        invoking it explicitly after induce() (or at any other point) if
        they want the extra refinement.
        """
        if self._current_heat is None:
            return

        self._current_heat = self._smooth_and_clean_heatmap(
            self._current_heat,
            step_percent=step_percent,
            max_iterations=max_iterations,
            max_cell_heat=max_cell_heat,
        )
        self._notify_change()

    @staticmethod
    def _smooth_and_clean_heatmap(
        heat_grid: np.ndarray,
        step_percent: float = 0.01,
        max_iterations: int = 150,
        max_cell_heat: float = 1.0,
    ) -> np.ndarray:
        if heat_grid is None or heat_grid.size == 0 or heat_grid.shape[0] < 3 or heat_grid.shape[1] < 3:
            return heat_grid

        grid = heat_grid.astype(np.float32, copy=True)
        h, w = grid.shape

        grid_max = float(np.max(grid))
        cap = max(max_cell_heat, grid_max) if grid_max > 0 else max_cell_heat

        for _ in range(max_iterations):
            active_mask = grid > 1e-6
            if not np.any(active_mask):
                break

            num_labels, labels = cv2.connectedComponents(active_mask.astype(np.uint8), connectivity=8)

            island_max_map = np.zeros_like(grid, dtype=np.float32)
            for label in range(1, num_labels):
                island_mask = (labels == label)
                i_max = float(np.max(grid[island_mask])) if np.any(island_mask) else 0.0
                island_max_map[island_mask] = i_max

            grid_next = grid.copy()
            transfers = 0

            ys, xs = np.nonzero(active_mask)
            for y, x in zip(ys, xs):
                val = grid[y, x]
                if val <= 1e-6:
                    continue

                i_max = island_max_map[y, x]
                if i_max <= 1e-6:
                    continue

                rel_heat_ratio = val / i_max
                dynamic_flow_scale = rel_heat_ratio * rel_heat_ratio

                min_y, max_y = max(0, y - 1), min(h, y + 2)
                min_x, max_x = max(0, x - 1), min(w, x + 2)

                highest_heat_val = -1.0
                highest_heat_coord = None
                neighbors = []

                for ny in range(min_y, max_y):
                    for nx in range(min_x, max_x):
                        if ny == y and nx == x:
                            continue
                        n_val = grid[ny, nx]
                        neighbors.append((ny, nx, n_val))
                        if n_val > highest_heat_val:
                            highest_heat_val = n_val
                            highest_heat_coord = (ny, nx)

                if highest_heat_coord is None or not neighbors:
                    continue

                dy = np.sign(highest_heat_coord[0] - y)
                dx = np.sign(highest_heat_coord[1] - x)

                inward_candidates = []
                for ny, nx, n_val in neighbors:
                    dot = (ny - y) * dy + (nx - x) * dx
                    if dot > 0:
                        inward_candidates.append((ny, nx, n_val))

                if not inward_candidates:
                    inward_candidates = neighbors

                target_coord = None
                min_cand_val = float("inf")
                for ny, nx, n_val in inward_candidates:
                    if grid_next[ny, nx] < cap and n_val < min_cand_val:
                        min_cand_val = n_val
                        target_coord = (ny, nx)

                if target_coord is not None and min_cand_val < val:
                    ty, tx = target_coord
                    available_capacity = cap - grid_next[ty, tx]
                    if available_capacity > 1e-6:
                        amount = min(val * step_percent * dynamic_flow_scale, available_capacity)
                        if amount > 1e-7:
                            grid_next[y, x] -= amount
                            grid_next[ty, tx] += amount
                            transfers += 1

            grid = grid_next
            if transfers == 0:
                break

        return grid.astype(heat_grid.dtype, copy=False)

    def induce(self) -> None:
        """Executes the complete heat creation lifecycle synchronously."""
        self.form_centroid_heatmap()
        self.dissipate_heat(phase="pre_enrich")
        self.enrich_heat()
        self.dissipate_heat(phase="post_enrich")
