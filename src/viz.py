# src/viz.py
from __future__ import annotations

import numpy as np


def to_grayscale(rgb: np.ndarray) -> np.ndarray:
   
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H,W,3), got {rgb.shape}")

    gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def reconstruct_image(patches: list[np.ndarray], rows: int, cols: int) -> np.ndarray:
    
    if not patches:
        raise ValueError("patches cannot be empty")

    ph, pw, ch = patches[0].shape
    out = np.zeros((rows * ph, cols * pw, ch), dtype=np.uint8)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * ph, (r + 1) * ph
            x0, x1 = c * pw, (c + 1) * pw
            out[y0:y1, x0:x1] = patches[idx]
            idx += 1

    return out


def changed_color_unchanged_gray(
    patches_rgb: list[np.ndarray],
    changed_indices: set[int] | list[int],
    rows: int,
    cols: int,
) -> np.ndarray:
    
    changed_set = set(changed_indices)

    vis_patches: list[np.ndarray] = []
    for i, patch in enumerate(patches_rgb):
        vis_patches.append(patch if i in changed_set else to_grayscale(patch))

    return reconstruct_image(vis_patches, rows, cols)
