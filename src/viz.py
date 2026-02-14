from pathlib import Path
import os
import numpy as np
import cv2

from io_utils import read_single_band_tif
from tiling import tile_image
from descriptor import DescriptorConfig
from change_detect import run_edge_merkle_change_detection
from viz import (
    changed_mask_from_indices,
    save_change_grid_png,
    save_overlay_on_image,
)

DATA_ROOT = Path("data/OSCD")
T1_PATH = DATA_ROOT / "imgs_1_rect" / "B04.tif"
T2_PATH = DATA_ROOT / "imgs_2_rect" / "B04.tif"

TILE_SIZE = 64


def _normalize_for_registration(img: np.ndarray) -> np.ndarray:
    """Robust normalize to float32 [0,1] using percentiles (works for uint16 GeoTIFF bands)."""
    x = img.astype(np.float32)
    lo, hi = np.percentile(x, (2.0, 98.0))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    x = (x - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _align_t2_to_t1(t1: np.ndarray, t2: np.ndarray, downsample: int = 4) -> tuple[np.ndarray, tuple[float, float], float]:
    """
    Estimate translation shift using phase correlation and warp t2 to align with t1.
    Returns: aligned_t2, (dx, dy), response
    """
    a = _normalize_for_registration(t1)
    b = _normalize_for_registration(t2)

    # Optional blur to reduce noise before registration
    a = cv2.GaussianBlur(a, (5, 5), 0)
    b = cv2.GaussianBlur(b, (5, 5), 0)

    if downsample > 1:
        a_small = cv2.resize(a, (a.shape[1] // downsample, a.shape[0] // downsample), interpolation=cv2.INTER_AREA)
        b_small = cv2.resize(b, (b.shape[1] // downsample, b.shape[0] // downsample), interpolation=cv2.INTER_AREA)
    else:
        a_small, b_small = a, b

    (dx, dy), response = cv2.phaseCorrelate(a_small, b_small)

    # Scale shift back to full-res pixels
    dx *= downsample
    dy *= downsample

    # Warp b (t2) by (-dx, -dy) to align to a (t1)
    M = np.array([[1.0, 0.0, -dx],
                  [0.0, 1.0, -dy]], dtype=np.float32)

    aligned_t2 = cv2.warpAffine(
        t2,
        M,
        dsize=(t2.shape[1], t2.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    return aligned_t2, (dx, dy), float(response)


def main() -> None:
    t1 = read_single_band_tif(T1_PATH)
    t2 = read_single_band_tif(T2_PATH)

    if t1.shape != t2.shape:
        raise ValueError(f"Shape mismatch t1={t1.shape}, t2={t2.shape}")

    # NEW: align t2 to t1
    t2_aligned, (dx, dy), resp = _align_t2_to_t1(t1, t2, downsample=4)

    tiles1, rows, cols = tile_image(t1, TILE_SIZE)
    tiles2, rows2, cols2 = tile_image(t2_aligned, TILE_SIZE)

    if (rows, cols) != (rows2, cols2):
        raise ValueError("Tile grid mismatch")

    cfg = DescriptorConfig(
        kind="sobel_mag",
        clip_percentiles=(2.0, 98.0),
        blur_ksize=5,
        canny_low=50,
        canny_high=150,
        downsample_to=(8, 8),
        quant_step=64,
    )

    result = run_edge_merkle_change_detection(tiles1, tiles2, cfg)

    os.makedirs("outputs", exist_ok=True)

    grid_mask = changed_mask_from_indices(result.changed_indices, (rows, cols))
    save_change_grid_png(grid_mask, "outputs/change_grid.png", scale=50)

    save_overlay_on_image(
        base_img=t2_aligned,   # overlay on aligned image
        grid_mask=grid_mask,
        tile_size=TILE_SIZE,
        out_path="outputs/overlay_t2_aligned.png",
        alpha=0.35,
    )

    before = len(result.changed_indices)

    print(f"registration_shift_px dx={dx:.2f}, dy={dy:.2f}, response={resp:.4f}")
    print(f"grid={rows}x{cols}")
    print(f"changed_tiles={before}/{rows*cols}")
    print("Saved outputs/change_grid.png and outputs/overlay_t2_aligned.png")


if __name__ == "__main__":
    main()
