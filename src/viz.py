import numpy as np
import cv2
from typing import List, Tuple

import numpy as np
import cv2
from typing import List, Tuple


def changed_mask_from_indices(changed_indices: List[int], grid_hw: Tuple[int, int]) -> np.ndarray:
   
    rows, cols = grid_hw
    mask = np.zeros((rows, cols), dtype=np.uint8)
    for idx in changed_indices:
        r = idx // cols
        c = idx % cols
        if 0 <= r < rows and 0 <= c < cols:
            mask[r, c] = 255
    return mask


def save_change_grid_png(mask_grid: np.ndarray, out_path: str, scale: int = 40) -> None:
   
  
    img = mask_grid.copy()
    img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(out_path, img)


def save_overlay_on_image(
    base_img: np.ndarray,
    grid_mask: np.ndarray,
    tile_size: int,
    out_path: str,
    alpha: float = 0.35
) -> None:
   
    if base_img.ndim == 2:
        base_bgr = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    else:
        base_bgr = base_img.copy()

    rows, cols = grid_mask.shape
    overlay = base_bgr.copy()

    for r in range(rows):
        for c in range(cols):
            if grid_mask[r, c] == 255:
                y0, y1 = r * tile_size, (r + 1) * tile_size
                x0, x1 = c * tile_size, (c + 1) * tile_size
                # red rectangle fill
                overlay[y0:y1, x0:x1] = (0, 0, 255)

    blended = cv2.addWeighted(overlay, alpha, base_bgr, 1.0 - alpha, 0)
    cv2.imwrite(out_path, blended)
