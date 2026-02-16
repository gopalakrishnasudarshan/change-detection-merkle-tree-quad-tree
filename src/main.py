# src/main.py
from pathlib import Path
from PIL import Image
import numpy as np

from src.tiling import tile_image_level
from src.hashing import patch_phash_bytes
from src.merkle import build_merkle_tree, merkle_diff_changed_leaves
from src.viz import changed_color_unchanged_gray

IMG1_PATH = Path("data/Image_11.png")
IMG2_PATH = Path("data/Image_22.png")
OUTPUT_PATH = Path("data/output_11.jpg")

# Sanity-check outputs
RAW_DIFF_PATH = Path("data/raw_diff_11.jpg")

LEVEL = 3
HASH_SIZE = 8


def load_rgb(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def save_raw_diff(img1: np.ndarray, img2: np.ndarray, out_path: Path) -> float:
    
    diff = np.abs(img1.astype(np.int16) - img2.astype(np.int16))  # HxWx3
    mean_diff = float(diff.mean())

    # For visualization: clamp to [0,255] and save as uint8 RGB diff image
    diff_u8 = np.clip(diff, 0, 255).astype(np.uint8)
    Image.fromarray(diff_u8).save(out_path)

    return mean_diff


def shift_sanity(img1: np.ndarray, img2: np.ndarray) -> None:
   
    base = float(np.abs(img1.astype(np.int16) - img2.astype(np.int16)).mean())

    left1 = float(np.abs(img1.astype(np.int16) - np.roll(img2, shift=-1, axis=1).astype(np.int16)).mean())
    right1 = float(np.abs(img1.astype(np.int16) - np.roll(img2, shift=+1, axis=1).astype(np.int16)).mean())
    up1 = float(np.abs(img1.astype(np.int16) - np.roll(img2, shift=-1, axis=0).astype(np.int16)).mean())
    down1 = float(np.abs(img1.astype(np.int16) - np.roll(img2, shift=+1, axis=0).astype(np.int16)).mean())

    print("\nAlignment sanity (mean abs diff):")
    print(f"  no shift : {base:.4f}")
    print(f"  x -1     : {left1:.4f}")
    print(f"  x +1     : {right1:.4f}")
    print(f"  y -1     : {up1:.4f}")
    print(f"  y +1     : {down1:.4f}")

    best = min(base, left1, right1, up1, down1)
    if best < base * 0.90:
        print("  NOTE: A 1px shift reduces diff noticeably -> likely slight mis-crop/misalignment.")


def main() -> None:
    img1 = load_rgb(IMG1_PATH)
    img2 = load_rgb(IMG2_PATH)

    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: img1={img1.shape} img2={img2.shape}")

    # --- Sanity checks (cropping/alignment) ---
    mean_diff = save_raw_diff(img1, img2, RAW_DIFF_PATH)
    print("Mean absolute pixel difference:", f"{mean_diff:.4f}")
    print("Saved raw diff image to:", RAW_DIFF_PATH)

    shift_sanity(img1, img2)

    # --- Main pipeline ---
    patches1, rows, cols = tile_image_level(img1, LEVEL, pad=False)
    patches2, _, _ = tile_image_level(img2, LEVEL, pad=False)

    leaf1 = [patch_phash_bytes(p, hash_size=HASH_SIZE) for p in patches1]
    leaf2 = [patch_phash_bytes(p, hash_size=HASH_SIZE) for p in patches2]

    tree1 = build_merkle_tree(leaf1)
    tree2 = build_merkle_tree(leaf2)

    root1 = tree1[-1][0]
    root2 = tree2[-1][0]

    changed = merkle_diff_changed_leaves(tree1, tree2)

    print("\nMerkle roots equal?", root1 == root2)
    print("root1:", root1.hex())
    print("root2:", root2.hex())

    print(f"\nMerkle changed leaves: {len(changed)}/{len(leaf1)}")
    print("changed_indices (first 50):", changed[:50])

    if changed:
        print("\nFirst changed (index -> (row,col)):")
        for i in changed[:10]:
            r = i // cols
            c = i % cols
            print(f"{i} -> ({r},{c})")

    output_img = changed_color_unchanged_gray(
        patches_rgb=patches2,
        changed_indices=changed,
        rows=rows,
        cols=cols,
    )

    Image.fromarray(output_img).save(OUTPUT_PATH)
    print("\nSaved visualization to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
