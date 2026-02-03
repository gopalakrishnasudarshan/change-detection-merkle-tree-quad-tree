from pathlib import Path
import os

from io_utils import read_single_band_tif
from tiling import tile_image
from descriptor import DescriptorConfig
from change_detect import run_edge_merkle_change_detection
from viz import changed_mask_from_indices, save_change_grid_png, save_overlay_on_image


DATA_ROOT = Path("data/OSCD")
T1_PATH = DATA_ROOT / "imgs_1_rect" / "B04.tif"
T2_PATH = DATA_ROOT / "imgs_2_Rect" / "B04.tif"

TILE_SIZE = 64


def main() -> None:
    
    t1 = read_single_band_tif(T1_PATH)
    t2 = read_single_band_tif(T2_PATH)

    if t1.shape != t2.shape:
        raise ValueError(f"Shape mismatch t1={t1.shape}, t2={t2.shape}")

  
    tiles1, rows, cols = tile_image(t1, TILE_SIZE)
    tiles2, rows2, cols2 = tile_image(t2, TILE_SIZE)

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
        base_img=t2,
        grid_mask=grid_mask,
        tile_size=TILE_SIZE,
        out_path="outputs/overlay_t2.png",
        alpha=0.35,
    )

    print(f"grid={rows}x{cols}")
    print(f"changed_tiles={len(result.changed_indices)}/{rows*cols}")
    print("Saved outputs/change_grid.png and outputs/overlay_t2.png")


if __name__ == "__main__":
    main()
