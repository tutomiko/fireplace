import logging
import time
import numpy as np
from typing import Callable, Optional

from .heatmap import FireplaceHeatmap

logger = logging.getLogger("fireplace.convergence")

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

        # Apply the exact min-max normalization logic from the orchestrator's payload formatting
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

    def dissipate_heat(self) -> None:
        """
        Progressively elevates the floor cutoff so cooler heat regions dissolve away
        across multiple simulated steps.
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

    def induce(self) -> None:
        """Executes the complete heat creation lifecycle synchronously."""
        self.form_centroid_heatmap()
        # Orchestrator runs dissipation prior to enrichment to cull floor heat first
        self.dissipate_heat()
        self.enrich_heat()
        self.dissipate_heat()
