import json
import logging
import queue
import threading
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from matplotlib.path import Path as MplPath
from PIL import Image
from pydantic import BaseModel
from transformers import AutoImageProcessor, AutoModel

from fireplace import Fireplace

HARNESS_DIR = Path(__file__).parent

logging.basicConfig(level=logging.INFO, format="%(message)s")

app = FastAPI()

# DINOv2 ViT input tile size and patch size (facebook/dinov2-base).
DINOV2_TILE_SIZE = 224
DINOV2_PATCH_SIZE = 14
DINOV2_EMBED_DIM = 768  # dinov2-base

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_processor = None
_model = None


def _load_model():
    global _processor, _model
    if _model is None:
        # auto-downloads weights on first call
        _processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        _model = AutoModel.from_pretrained("facebook/dinov2-base")
        _model.eval()
    return _processor, _model


def _tile_image(image_rgb: np.ndarray, tile_size: int):
    """Splits the image into non-overlapping tile_size x tile_size tiles."""
    h, w = image_rgb.shape[:2]
    cols = w // tile_size
    rows = h // tile_size

    tiles = []
    for row in range(rows):
        for col in range(cols):
            x0, y0 = col * tile_size, row * tile_size
            crop = image_rgb[y0:y0 + tile_size, x0:x0 + tile_size]
            tiles.append((crop, x0, y0))
    return tiles


def _tiles_to_tensor(tiles) -> torch.Tensor:
    batch = np.stack([t[0] for t in tiles], axis=0).astype(np.float32) / 255.0
    batch = (batch - _IMAGENET_MEAN) / _IMAGENET_STD
    batch = np.transpose(batch, (0, 3, 1, 2))
    return torch.from_numpy(batch)


@torch.no_grad()
def _embed_image(image_path: Path) -> np.ndarray:
    """Tiles the image at native resolution into 224x224 tiles, runs DINOv2
    per tile, and stitches the per-tile patch grids back together into one
    whole-image patch embedding grid."""
    _, model = _load_model()

    image = Image.open(image_path).convert("RGB")
    image_rgb = np.array(image)
    h, w = image_rgb.shape[:2]

    # Pad the image to a multiple of the tile size so we don't drop any valid data
    # at the right and bottom edges.
    pad_h = (DINOV2_TILE_SIZE - h % DINOV2_TILE_SIZE) % DINOV2_TILE_SIZE
    pad_w = (DINOV2_TILE_SIZE - w % DINOV2_TILE_SIZE) % DINOV2_TILE_SIZE
    if pad_h > 0 or pad_w > 0:
        image_rgb = np.pad(image_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')

    tiles = _tile_image(image_rgb, DINOV2_TILE_SIZE)
    if not tiles:
        raise ValueError(
            f"{image_path.name} is smaller than the {DINOV2_TILE_SIZE}x{DINOV2_TILE_SIZE} tile size"
        )

    patches_per_side = DINOV2_TILE_SIZE // DINOV2_PATCH_SIZE

    tensor = _tiles_to_tensor(tiles)
    outputs = model(pixel_values=tensor)
    # Drop the CLS token, keep the per-patch tokens.
    patch_tokens = outputs.last_hidden_state[:, 1:, :].reshape(
        len(tiles), patches_per_side, patches_per_side, DINOV2_EMBED_DIM
    ).numpy()

    padded_h, padded_w = image_rgb.shape[:2]
    patch_rows_total = (padded_h // DINOV2_TILE_SIZE) * patches_per_side
    patch_cols_total = (padded_w // DINOV2_TILE_SIZE) * patches_per_side
    whole = np.zeros((patch_rows_total, patch_cols_total, DINOV2_EMBED_DIM), dtype=patch_tokens.dtype)

    for (_, x0, y0), tile_patches in zip(tiles, patch_tokens):
        r0 = (y0 // DINOV2_TILE_SIZE) * patches_per_side
        c0 = (x0 // DINOV2_TILE_SIZE) * patches_per_side
        whole[r0:r0 + patches_per_side, c0:c0 + patches_per_side] = tile_patches

    return whole


def _ensure_embeddings() -> None:
    for name in ("left", "right"):
        npy_path = HARNESS_DIR / f"{name}.npy"
        if npy_path.exists():
            continue
        image_path = HARNESS_DIR / f"{name}.png"
        embedding = _embed_image(image_path)
        np.save(npy_path, embedding)


def _image_size(name: str) -> tuple[int, int]:
    with Image.open(HARNESS_DIR / f"{name}.png") as im:
        return im.size  # (width, height)


def _patches_in_polygon(grid_shape: tuple[int, int], image_size: tuple[int, int], polygon: list[list[float]]) -> np.ndarray:
    """Returns (row, col) indices of patch grid cells whose center falls
    inside the given polygon. `polygon` points and `image_size` are in the
    source image's natural pixel coordinates. `grid_shape` is (rows, cols)
    of the patch embedding grid, laid out over the same natural pixel
    extent covered by whole tiles (see _embed_image / _tile_image)."""
    grid_rows, grid_cols = grid_shape
    img_w, img_h = image_size

    # Natural-pixel extent actually covered by the patch grid (tiles are
    # dropped at the edges, so this can be smaller than the full image).
    covered_w = grid_cols * DINOV2_PATCH_SIZE
    covered_h = grid_rows * DINOV2_PATCH_SIZE

    poly = np.array(polygon, dtype=np.float64)
    path = MplPath(poly)

    centers_x = (np.arange(grid_cols) + 0.5) * DINOV2_PATCH_SIZE
    centers_y = (np.arange(grid_rows) + 0.5) * DINOV2_PATCH_SIZE
    grid_x, grid_y = np.meshgrid(centers_x, centers_y)  # each (grid_rows, grid_cols)
    centers = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    inside = path.contains_points(centers).reshape(grid_rows, grid_cols)
    rows, cols = np.nonzero(inside)
    return np.stack([rows, cols], axis=1)


@app.on_event("startup")
def on_startup():
    _ensure_embeddings()


@app.post("/embed")
def embed():
    results = {}
    for name in ("left", "right"):
        image_path = HARNESS_DIR / f"{name}.png"
        embedding = _embed_image(image_path)
        npy_path = HARNESS_DIR / f"{name}.npy"
        np.save(npy_path, embedding)
        results[name] = {"shape": list(embedding.shape), "saved_to": str(npy_path)}
    return results


class LassoRequest(BaseModel):
    # Each lasso is a list of [x, y] points in left.png's natural pixel coords.
    lassos: list[list[list[float]]]
    decay: float = 0.01
    decay_threshold: float = 0.2


def _upsample_grid_to_pixels(grid: np.ndarray, right_size: tuple[int, int], resample=Image.BILINEAR) -> np.ndarray:
    """Upsamples a patch-grid array (float 0..1 or bool) to the right
    image's pixel resolution, cropping off tile padding."""
    grid_rows, grid_cols = grid.shape
    covered_w = grid_cols * DINOV2_PATCH_SIZE
    covered_h = grid_rows * DINOV2_PATCH_SIZE

    src = (np.clip(grid.astype(np.float32), 0.0, 1.0) * 255).astype(np.uint8)
    img = Image.fromarray(src)
    img = img.resize((covered_w, covered_h), resample=resample)
    arr = np.array(img)

    return arr[:right_size[1], :right_size[0]]


def _mask_to_payload(fp_heatmap, right_size: tuple[int, int]) -> dict:
    """Builds the per-frame payload: just the continuous heat mask, upsampled
    to the right image's pixel resolution."""
    mask = fp_heatmap.get_mask()
    full_mask = _upsample_grid_to_pixels(mask, right_size, resample=Image.BILINEAR)

    return {
        "width": right_size[0],
        "height": right_size[1],
        "mask": full_mask.tolist(),
    }


@app.post("/heatmap")
def heatmap(request: LassoRequest):
    if not request.lassos:
        raise HTTPException(status_code=400, detail="No lassos provided")

    left_grid = np.load(HARNESS_DIR / "left.npy")
    right_grid = np.load(HARNESS_DIR / "right.npy")
    left_size = _image_size("left")
    right_size = _image_size("right")

    grid_shape = left_grid.shape[:2]

    centroids = []
    patch_embeddings = []
    for polygon in request.lassos:
        if len(polygon) < 3:
            continue
        cell_indices = _patches_in_polygon(grid_shape, left_size, polygon)
        if len(cell_indices) == 0:
            continue
        cell_vectors = left_grid[cell_indices[:, 0], cell_indices[:, 1]]
        centroids.append(cell_vectors.mean(axis=0))
        patch_embeddings.extend(cell_vectors)

    if not centroids:
        raise HTTPException(status_code=400, detail="No patches fell inside the provided lassos")

    frame_queue: "queue.Queue[dict | None]" = queue.Queue()

    def on_heatmap_changed(fp_heatmap):
        # This fires on every animation step and needs to stay cheap to keep
        # the stream smooth.
        payload = _mask_to_payload(fp_heatmap, right_size)
        frame_queue.put(payload)

    def run_induce():
        fp = Fireplace()
        fp.on_heatmap_changed = on_heatmap_changed
        fp.set_class_precomputed_centroids_provider(lambda: np.array(centroids))
        fp.set_class_nn_patch_embeddings_provider(lambda: np.array(patch_embeddings))
        fp.set_embedding_space(right_grid)
        fp.set_decay(request.decay)
        fp.set_decay_threshold(request.decay_threshold)
        fp.set_heatmap_rendering_clock(0.1)  # 100ms between frames
        fp.induce()
        frame_queue.put(None)  # sentinel: stream done

    thread = threading.Thread(target=run_induce, daemon=True)
    thread.start()

    def event_stream():
        while True:
            payload = frame_queue.get()
            if payload is None:
                yield "event: done\ndata: {}\n\n"
                break
            yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/")
def index():
    return FileResponse(HARNESS_DIR / "index.html")


app.mount("/static", StaticFiles(directory=HARNESS_DIR), name="static")
